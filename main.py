import cv2
import numpy as np
import onnxruntime as ort

import gestualis

# ==========================================
# 1. Configuration & Model Loading
# ==========================================
MODEL_PATH = "model/hand_landmark_detector.onnx" # Replace with your downloaded .onnx file

flag = 0

# Initialize ONNX Runtime session
# (Optional) You can specify providers like ['CUDAExecutionProvider'] for GPU
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

# Get input details dynamically
input_details = session.get_inputs()[0]
input_name = input_details.name
input_shape = input_details.shape  # Usually [1, 3, 256, 256] for PyTorch exports

print(f"Model expects input shape: {input_shape}")

# Determine if the model expects NCHW (Channels First) or NHWC (Channels Last)
if input_shape[1] == 3:
    is_nchw = True
    input_height, input_width = input_shape[2], input_shape[3]
else:
    is_nchw = False
    input_height, input_width = input_shape[1], input_shape[2]

# ==========================================
# 2. Video Capture
# ==========================================
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Mirror the frame and convert BGR (OpenCV default) to RGB
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ==========================================
    # 3. Pre-processing
    # ==========================================
    # Resize to the model's expected size
    resized_frame = cv2.resize(frame_rgb, (input_width, input_height))
    
    # Normalize pixel values to [0.0, 1.0] as float32
    input_data = resized_frame.astype(np.float32) / 255.0
    
    if is_nchw:
        # Convert from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
        input_data = np.transpose(input_data, (2, 0, 1))
        
    # Expand dimensions to add the batch size: shape becomes [1, C, H, W] or [1, H, W, C]
    input_data = np.expand_dims(input_data, axis=0)
    
    # ==========================================
    # 4. Inference
    # ==========================================
    # Run the ONNX model. None means "return all outputs".
    outputs = session.run(None, {input_name: input_data})
    
    # ==========================================
    # 5. Post-processing & Visualization
    # ==========================================
    # Search the outputs for the tensor containing exactly 63 values (21 landmarks * 3 coordinates)

    landmarks_tensor = None
    for out in outputs:
        if out.size == 63:
            landmarks_tensor = out.flatten() # Flatten just in case it has nested batch dims
            break
            
    if landmarks_tensor is not None:
        # Reshape to a 21x3 array
        h, w, _ = frame.shape
        landmarks = landmarks_tensor.reshape(21, 3)
        if flag == 1:
            dict_landmarks = {i: landmarks[i] for i in range(21)}

            quat, _ = gestualis.compute.calculate_required_rotation(landmarks[0], landmarks[9], landmarks[8])

            for i in range(len(landmarks)):
                landmarks[i] = gestualis.compute.apply_rotation(landmarks[i], quat, landmarks[0])
        elif flag == 2:
            landmarks = gestualis.compute.normalize_hand_orientation(landmarks, (h, w))


        
        for (x, y, z) in landmarks:
            # Check if coordinates are normalized (0.0 - 1.0) or raw pixel sizes relative to input
            if x < 2.0 and y < 2.0: 
                px = int(x * w)
                py = int(y * h)
            else:
                px = int((x / input_width) * w)
                py = int((y / input_height) * h)
                
            # Draw a circle for each landmark
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

    # Display the result
    cv2.imshow('MediaPipe Hand ONNX', frame)

    # Key inputs
    key = cv2.waitKey(1) & 0xFF
    
    # Break loop if 'q' is pressed
    if key == ord('q'):
        break
    if key == ord('s'):
        flag = (flag + 1) % 3

# Clean up
cap.release()
cv2.destroyAllWindows()

