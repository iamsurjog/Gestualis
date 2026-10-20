import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="model/hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.2,
    min_tracking_confidence=0.2,
)

FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]
FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky"]


def get_distance(p1, p2):
    """Calculates Euclidean distance between two MediaPipe landmarks in 3D."""
    return math.sqrt(
        (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2
    )


cap = cv2.VideoCapture(0)

with vision.HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            for idx, landmarks in enumerate(detection_result.hand_landmarks):
                handedness = "Right"
                if detection_result.handedness and idx < len(
                    detection_result.handedness
                ):
                    handedness = (
                        detection_result.handedness[idx][0].category_name
                    )

                wrist = landmarks[0]
                pinky_mcp = landmarks[17]
                thumb_tip = landmarks[4]
                thumb_ip = landmarks[3]

                open_fingers = []

                # --- Thumb Logic (Independent of Hand Tilt) ---
                # Compares distance from thumb tip to pinky MCP base
                dist_thumb_tip_to_pinky = get_distance(thumb_tip, pinky_mcp)
                dist_thumb_ip_to_pinky = get_distance(thumb_ip, pinky_mcp)

                if dist_thumb_tip_to_pinky > dist_thumb_ip_to_pinky:
                    open_fingers.append("Thumb")

                # --- 4 Fingers Logic (Independent of Hand Tilt) ---
                # Tip must be farther from the wrist than the PIP joint
                for tip_idx, pip_idx, name in zip(
                    FINGER_TIPS, FINGER_PIPS, FINGER_NAMES
                ):
                    dist_tip_to_wrist = get_distance(landmarks[tip_idx], wrist)
                    dist_pip_to_wrist = get_distance(landmarks[pip_idx], wrist)

                    # Multiplier (1.1) adds a margin of error against curled finger tips
                    if dist_tip_to_wrist > dist_pip_to_wrist * 1.1:
                        open_fingers.append(name)

                # Visual overlay
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                status = f"{handedness}: {', '.join(open_fingers) if open_fingers else 'Fist'}"
                cv2.putText(
                    frame,
                    status,
                    (20, 40 + idx * 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("Hand Tracking (Rotation Invariant)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
