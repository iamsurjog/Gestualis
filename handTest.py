import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import gestualis

# ==========================================
# 1. Configuration & Model Loading
# ==========================================
MODEL_PATH = "model/hand_landmarker.task"


# Configure the MediaPipe Hand Landmarker options
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # Highly recommended for webcam streams
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# 2. Video Capture
# ==========================================
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

# VIDEO running mode requires a strictly increasing timestamp
start_time = time.time()

# Create the detector using a context manager so it safely cleans up memory
with vision.HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Mirror the frame and convert BGR (OpenCV) to RGB (MediaPipe expects RGB)
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ==========================================
        # 3. Pre-processing & Inference
        # ==========================================
        # Convert the numpy array into MediaPipe's Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Calculate milliseconds since the loop started
        timestamp_ms = int((time.time() - start_time) * 1000)
        
        # Run inference
        results = detector.detect_for_video(mp_image, timestamp_ms)
        
        # ==========================================
        # 4. Post-processing & Visualization
        # ==========================================
        # Check if any hands were detected in this frame
        if results.hand_landmarks:
            # Grab the first detected hand
            hand_landmarks = results.hand_landmarks[0]
            
            # Convert MediaPipe's NormalizedLandmark objects into a 21x3 numpy array
            # so your existing gestualis rotation logic works without modification.
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
            

            h, w, _ = frame.shape
            dict_landmarks = {i: landmarks[i] for i in range(21)}
            hand = gestualis.compute.simple_hand(dict_landmarks)

            # s = gestualis.store.compare(hand)
            # print(s)

            # gestualis.store.store(hand, name="10.dat")
            # break

            # print(landmarks)

            
            # print(landmarks)
            for (x, y, z) in landmarks:
                # MediaPipe coordinates are strictly normalized from 0.0 to 1.0
                px = int(x * w)
                py = int(y * h)
                
                # Draw a circle for each landmark
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

        # Display the result
        cv2.imshow('MediaPipe Hand Tasks API', frame)

        # Key inputs
        key = cv2.waitKey(1) & 0xFF
        
        # Break loop if 'q' is pressed
        if key == ord('q'):
            break
        # Toggle flag if 's' is pressed

# Clean up
cap.release()
cv2.destroyAllWindows()
