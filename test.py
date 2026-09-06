import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# STEP 1: Create MediaPipe detector
# -----------------------------

base_options = python.BaseOptions(
    model_asset_path=r'C:\Users\Sukruta Nadkarni\OneDrive - vit.ac.in\Desktop\Project - I\Gestualis\Gestualis-v1-dev\models\hand_landmarker.task'
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)


# -----------------------------
# STEP 2: Open webcam
# -----------------------------

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Could not open webcam")
    exit()


# MediaPipe hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# -----------------------------
# STEP 3: Process webcam frames
# -----------------------------

while True:

    ret, frame = cam.read()

    if not ret:
        print("Failed to grab frame")
        break

    # OpenCV gives BGR
    # MediaPipe expects RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert frame to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
    )

    # -----------------------------
    # THIS IS THE IMPORTANT PART
    # Actually run the detector
    # -----------------------------
    detection_result = detector.detect(mp_image)

    # -----------------------------
    # Draw detected hands
    # -----------------------------

    if detection_result.hand_landmarks:

        h, w, _ = frame.shape

        for hand_landmarks in detection_result.hand_landmarks:

            pixel_points = {}

            for idx, landmark in enumerate(hand_landmarks):

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                pixel_points[idx] = (x, y)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (255, 0, 0),
                    cv2.FILLED
                )

            # Draw hand skeleton
            for start_idx, end_idx in HAND_CONNECTIONS:

                if start_idx in pixel_points and end_idx in pixel_points:

                    cv2.line(
                        frame,
                        pixel_points[start_idx],
                        pixel_points[end_idx],
                        (255, 255, 0),
                        2
                    )

    # Display webcam
    cv2.imshow("MediaPipe Hand Landmarker", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# -----------------------------
# STEP 4: Clean up
# -----------------------------

cam.release()
detector.close()
cv2.destroyAllWindows()