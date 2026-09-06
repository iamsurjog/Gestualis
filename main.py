import cv2
import gestualis

# Initialize the camera
cam = cv2.VideoCapture(0)

# Define how the joints connect to form a hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                                 # Wrist to Pinky base
]

print("Starting Gestualis hand tracker. Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect landmarks and get canonical (rotation-invariant) representation
    features, canonical_landmarks, raw_result = gestualis.get_points(img_rgb)

    # 1. Draw raw detected hand (green skeleton directly on user's hand)
    if raw_result and raw_result.hand_landmarks:
        for hand_lms in raw_result.hand_landmarks:
            raw_pts = {}
            for idx, lm in enumerate(hand_lms):
                rx, ry = int(lm.x * w), int(lm.y * h)
                raw_pts[idx] = (rx, ry)
                cv2.circle(frame, (rx, ry), 4, (0, 255, 0), cv2.FILLED)

            for s_idx, e_idx in HAND_CONNECTIONS:
                if s_idx in raw_pts and e_idx in raw_pts:
                    cv2.line(frame, raw_pts[s_idx], raw_pts[e_idx], (0, 200, 0), 2)

    # 2. Draw canonical normalized hand (cyan/blue skeleton centered on screen, stays upright)
    if canonical_landmarks:
        canonical_pts = {}
        for idx, coords in canonical_landmarks.items():
            # Shift by +0.5 to center origin (wrist) at screen center
            cx = int((coords[0] + 0.5) * w)
            cy = int((coords[1] + 0.5) * h)
            canonical_pts[idx] = (cx, cy)
            cv2.circle(frame, (cx, cy), 5, (255, 100, 0), cv2.FILLED)

        # Draw bones connecting the canonical joints
        for s_idx, e_idx in HAND_CONNECTIONS:
            if s_idx in canonical_pts and e_idx in canonical_pts:
                cv2.line(frame, canonical_pts[s_idx], canonical_pts[e_idx], (255, 255, 0), 2)

        # Add visual guide text
        cv2.putText(frame, "Canonical Hand (Normalized: Upright & Front-Facing)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Live Tracked Hand (Green)",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Gestualis Hand Canonicalizer", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources when done
cam.release()
gestualis.camera.close_detector()
cv2.destroyAllWindows()

