from typing import Callable
import numpy as np
import numpy.typing as npt

def calculate_required_rotation(
    wrist: npt.NDArray[np.float64], 
    p_axis: npt.NDArray[np.float64], 
    p_rotate: npt.NDArray[np.float64],
    threshold: float | None = None
) -> tuple[npt.NDArray[np.float64], float]:
    """
    Calculates the quaternion required to rotate a point around a spatial axis 
    until its Z-coordinate matches the axis target.

    This function extracts the Z-component of Rodrigues' Rotation Formula to 
    solve for the exact angle algebraically, avoiding iterative solvers. 
    It is optimized for real-time 3D landmark normalization.

    Parameters
    ----------
    wrist : npt.NDArray[np.float64]
        A (3,) array representing the 3D origin point (e.g., MediaPipe Landmark 0).
    p_axis : npt.NDArray[np.float64]
        A (3,) array representing the axis endpoint (e.g., Landmark 9). 
        The rotation axis is defined from `wrist` to `p_axis`.
    p_rotate : npt.NDArray[np.float64]
        A (3,) array representing the point to be rotated (e.g., Landmark 5).

    Returns
    -------
    quaternion : npt.NDArray[np.float64]
        A (4,) array containing the rotation quaternion [w, x, y, z].
    theta : float
        The rotation angle in radians in the range [-pi, pi]. 
        Useful for thresholding severe hand flips.

    Notes
    -----
    If the axis length is zero, or the target Z-depth is physically impossible 
    to reach via rotation, the function returns an identity quaternion 
    ([1.0, 0.0, 0.0, 0.0]) and an angle of 0.0.
    """
    

    # Translate everything so the wrist is at the origin (0,0,0)
    axis_vec: npt.NDArray[np.float64] = p_axis - wrist
    target_vec: npt.NDArray[np.float64]  = p_rotate - wrist
    
    # We want the rotated Point 5's Z to match Point 9's translated Z
    target_z = axis_vec[2]
    
    # Normalize the rotation axis (k)
    axis_len = np.linalg.norm(axis_vec)
    if axis_len == 0:
        return np.array([1.0, 0.0, 0.0, 0.0]), 0.0
    k = axis_vec / axis_len
    
    # Setup Rodrigues' rotation equation for the Z-axis: 
    # a*cos(theta) + b*sin(theta) = c
    P = target_vec
    D = np.dot(k, P)
    
    a = P[2] - k[2] * D
    b = k[0] * P[1] - k[1] * P[0]  # The Z-component of (k cross P)
    c = target_z - k[2] * D
    
    # Solve for theta
    R = np.hypot(a, b)
    if R == 0:
        return np.array([1.0, 0.0, 0.0, 0.0]), 0.0
        
    # np.clip prevents math domain errors if the exact Z is physically 
    # out of reach due to the radius of the finger
    ratio = np.clip(c / R, -1.0, 1.0)
    
    alpha = np.arctan2(b, a)
    acos_val = np.arccos(ratio)
    
    # The circle intersects the Z-plane twice, giving two possible angles
    theta1 = alpha + acos_val
    theta2 = alpha - acos_val
    
    # Wrap angles to [-pi, pi] to find the shortest rotation path
    theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
    theta2 = (theta2 + np.pi) % (2 * np.pi) - np.pi
    
    # Pick the smallest rotation magnitude
    theta = theta1 if abs(theta1) < abs(theta2) else theta2

    # Theta threshold check:
    if threshold is not None:
        if theta > threshold:
            return np.array([1.0, 0.0, 0.0, 0.0]), theta
    
    # 5. Convert Axis-Angle to a Quaternion [w, x, y, z]
    half_theta = theta / 2.0
    sin_half = np.sin(half_theta)
    
    w = np.cos(half_theta)
    x = k[0] * sin_half
    y = k[1] * sin_half
    z = k[2] * sin_half
    
    return np.array([w, x, y, z]), theta


def apply_rotation(point: npt.NDArray[np.float64],
                   quaternion: npt.NDArray[np.float64],
                   origin: npt.NDArray[np.float64]
):
    """
    Applies a quaternion rotation to an array of (N, 3) landmarks.
    """
    w, x, y, z = quaternion
    
    # Convert quaternion to a 3x3 rotation matrix
    rot_matrix = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,         1 - 2*x*x - 2*y*y]
    ])

    # Shift the hand so the wrist is at the origin (0,0,0)
    shifted_landmark = point - origin

    # Rotate all 21 points simultaneously using matrix multiplication
    rotated_landmark = shifted_landmark @ rot_matrix.T

    # Shift the hand back to its original world position
    #final_landmark = rotated_landmark + origin

    # NOTE: replace with final landmark if necessary

    return rotated_landmark


def canonicalize_landmarks(
    landmarks: dict[int, npt.NDArray[np.float64]],
    normalize_scale: bool = False,
    reference_palm_len: float = 0.22,
) -> dict[int, npt.NDArray[np.float64]]:
    """
    Rotates and translates the 3D hand landmarks into a canonical reference frame:
    - Origin (0, 0, 0) is at the Wrist (Landmark 0).
    - Longitudinal axis: Wrist to Middle MCP (Landmark 9) points straight UP (negative Y on screen).
    - Palm normal: Faces the front (camera).
    - Lateral axis: Spans across the knuckles (Index MCP 5 to Pinky MCP 17).

    The resulting canonical coordinates are 100% invariant to hand rotation (pitch, roll, yaw)
    and translation in 3D space. When the user rotates their hand, the canonical landmarks
    remain stable in the standard front-facing upright pose.

    Parameters
    ----------
    landmarks : dict[int, npt.NDArray[np.float64]]
        Dictionary of landmark index to 3D coordinate array [x, y, z].
    normalize_scale : bool, default False
        If True, normalizes the hand size so distance from camera does not affect landmark values.
    reference_palm_len : float, default 0.22
        Reference palm length (distance from Wrist 0 to Middle MCP 9) when scale normalization is active.

    Returns
    -------
    canonical : dict[int, npt.NDArray[np.float64]]
        Dictionary of canonicalized 3D landmark coordinates.
    """
    if not landmarks or 0 not in landmarks or 9 not in landmarks:
        return {}

    p0 = landmarks[0]
    p9 = landmarks[9]

    # Up vector (wrist to middle MCP)
    u_vec = p9 - p0
    u_len = float(np.linalg.norm(u_vec))
    if u_len < 1e-7:
        return {k: v - p0 for k, v in landmarks.items()}

    u_hat = u_vec / u_len

    # Transverse / knuckle vector (pinky to index)
    if 5 in landmarks and 17 in landmarks:
        p5 = landmarks[5]
        p17 = landmarks[17]
        v_vec = p5 - p17
        # Orthogonalize v with respect to u
        v_perp = v_vec - np.dot(v_vec, u_hat) * u_hat
        v_len = float(np.linalg.norm(v_perp))
        if v_len > 1e-7:
            v_hat = v_perp / v_len
        else:
            v_hat = np.array([1.0, 0.0, 0.0])
    else:
        # Fallback if 5 and 17 are not available
        v_hat = np.array([1.0, 0.0, 0.0])

    # Normal vector perpendicular to palm: n = v x u
    n_hat = np.cross(v_hat, u_hat)
    n_len = float(np.linalg.norm(n_hat))
    if n_len > 1e-7:
        n_hat = n_hat / n_len
    else:
        n_hat = np.array([0.0, 0.0, 1.0])

    # Re-orthogonalize v_hat to ensure strict right-handed orthonormal basis
    v_hat = np.cross(u_hat, n_hat)

    # Scale factor
    scale = (reference_palm_len / u_len) if (normalize_scale and u_len > 1e-7) else 1.0

    canonical = {}
    for idx, pt in landmarks.items():
        p_rel = pt - p0
        cx = float(np.dot(p_rel, v_hat) * scale)
        cy = float(np.dot(p_rel, u_hat) * scale)
        cz = float(np.dot(p_rel, n_hat) * scale)
        # In screen/camera coordinates: X is right, Y is up (negative screen y), Z is depth
        canonical[idx] = np.array([cx, -cy, cz], dtype=np.float64)

    return canonical


def simple_hand(landmarks: dict[int, npt.NDArray[np.float64]]) -> tuple[float, float, float, float, float]:
    """Computes invariant finger lengths from canonicalized landmarks."""
    if not landmarks or len(landmarks) < 21:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    canonical = canonicalize_landmarks(landmarks)
    thumb = float(np.linalg.norm(canonical[4] - canonical[1]))
    index = float(np.linalg.norm(canonical[8] - canonical[5]))
    middle = float(np.linalg.norm(canonical[12] - canonical[9]))
    ring = float(np.linalg.norm(canonical[16] - canonical[13]))
    little = float(np.linalg.norm(canonical[20] - canonical[17]))
    return thumb, index, middle, ring, little


def calculate_angle(vec1: npt.NDArray[np.float64], vec2: npt.NDArray[np.float64]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors not of same length")
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot = np.dot(vec1, vec2)
    return float(np.clip(dot / (norm1 * norm2), -1.0, 1.0))


def hand_comparator(hand1: list[npt.NDArray[np.float64]],
                  hand2: list[npt.NDArray[np.float64]],
                  innerOperator: Callable[[float], float] = lambda x: x,
                  outerOperator: Callable[[list[float]], float] = sum,
                  finalOperator: Callable[[float], float] = lambda x: x
                  ) -> float:
    angles = []
    for i in range(len(hand1)):
        angle = innerOperator(calculate_angle(hand1[i], hand2[i]))
        angles.append(angle)
        
    single_value = outerOperator(angles)
    
    return finalOperator(single_value)


def get_all(landmarks: dict[int, npt.NDArray[np.float64]]) -> dict[int, npt.NDArray[np.float64]]:
    """
    Returns all 21 hand landmarks rotated back into canonical front-facing upright pose.
    Invariant to 3D rotation and translation.
    """
    if not landmarks:
        return {}
    return canonicalize_landmarks(landmarks)

