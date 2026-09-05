import cv2
import mediapipe as mp
import gestualis 

# Initialize the camera
cam = cv2.VideoCapture(0)

# 1. Define how the joints connect to form a hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                                 # Wrist to Pinky base
]

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Call your function ONCE per frame
    hand, rotated_landmarks, raw_result = gestualis.get_points(img_rgb)

    if rotated_landmarks:
        h, w, c = frame.shape
        pixel_points = {}
        
        # 2. Convert your rotated normalized coordinates to pixel coordinates
        for idx, coords in rotated_landmarks.items():
            
            # Shift X and Y by +0.5 to move the origin to the center of the screen
            cx = int((coords[0] + 0.5) * w)
            cy = int((coords[1] + 0.5) * h)
            
            pixel_points[idx] = (cx, cy)
            
            # Draw a blue circle for the rotated joints
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            
            # --- DEBUGGING PRINT ---
            # Print the exact pixel coordinates for the Wrist (0) and Middle finger (9)
            if idx in [0, 9]:
                print(f"Point {idx} attempting to draw at pixels: X={cx}, Y={cy}")

        # Draw the "bones" connecting the rotated joints
        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            if start_idx in pixel_points and end_idx in pixel_points:
                cv2.line(frame, pixel_points[start_idx], pixel_points[end_idx], (255, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Webcam", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources when done
cam.release()
cv2.destroyAllWindows()
