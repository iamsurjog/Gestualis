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
    # final_landmark = rotated_landmark + wrist_pos

    # NOTE: replace with final landmark if necessary

    return rotated_landmark


def simple_hand(landmarks):
    quat, theta = calculate_required_rotation(wrist=landmarks[0], p_axis=landmarks[9], p_rotate=landmarks[8])
    rotated = {}
    rotated[9] = landmarks[9]
    rotated[0] = landmarks[0]
    for i in [1, 4, 5, 8, 12, 13, 16, 17, 20]:
        rotated[i] = apply_rotation(point=landmarks[i], quaternion=quat, origin=landmarks[0])
    
    thumb = np.linalg.norm(rotated[4] - rotated[1])
    index = np.linalg.norm(rotated[8] - rotated[5])
    middle = np.linalg.norm(rotated[12] - rotated[9])
    ring = np.linalg.norm(rotated[16] - rotated[13])
    little = np.linalg.norm(rotated[20] - rotated[17])
    return thumb, index, middle, ring, little


def calculate_angle(vec1: npt.NDArray[np.float64], vec2: npt.NDArray[np.float64]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors not of same length")
    dot = np.dot(vec1, vec2)
    return np.clip(dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0)


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


def test():
    print("TEST WORKS")
