import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import onnxruntime as ort

import gestualis


def get_hand_landmarks_onnx(image_path, model_path="model/hand_landmark_detector.onnx"):
    """
    Detects hand landmarks in a static image using an ONNX model and 
    formats them for gestualis.
    
    Returns:
        A list of dictionaries. Each dictionary represents a hand, mapping 
        landmark indices (0-20) to a numpy array of [x, y, z].
    """
    # 1. Initialize ONNX session
    # (Note: For bulk processing, you should initialize this outside the function to save time)
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    input_details = session.get_inputs()[0]
    input_name = input_details.name
    input_shape = input_details.shape
    
    # Determine model input format
    if input_shape[1] == 3:
        is_nchw = True
        input_height, input_width = input_shape[2], input_shape[3]
    else:
        is_nchw = False
        input_height, input_width = input_shape[1], input_shape[2]

    # 2. Read and Pre-process the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return []

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize
    resized_image = cv2.resize(image_rgb, (input_width, input_height))
    input_data = resized_image.astype(np.float32) / 255.0
    
    if is_nchw:
        input_data = np.transpose(input_data, (2, 0, 1))
        
    input_data = np.expand_dims(input_data, axis=0)

    # 3. Inference
    outputs = session.run(None, {input_name: input_data})

    # 4. Post-processing
    landmarks_tensor = None
    for out in outputs:
        if out.size == 63:
            landmarks_tensor = out.flatten()
            break
            
    if landmarks_tensor is not None:
        # Reshape to 21x3
        landmarks = landmarks_tensor.reshape(21, 3)
        
        # Check if coordinates need normalization (if they are absolute pixel values > 2.0)
        # We normalize them to 0.0 - 1.0 to match MediaPipe's standard output behavior.
        if landmarks[0][0] >= 2.0:
            landmarks[:, 0] = landmarks[:, 0] / input_width   # X
            landmarks[:, 1] = landmarks[:, 1] / input_height  # Y
            landmarks[:, 2] = landmarks[:, 2] / input_width   # Z 

        # Convert to dictionary {0: [x,y,z], 1: [x,y,z], ...} for gestualis
        dict_landmarks = {i: landmarks[i] for i in range(21)}
        
        # Return as a list of hands to match the previous function's signature
        return [dict_landmarks]

    return []


def get_hand_landmarks_mediapipe(image_path, model_path="model/hand_landmarker.task"):
    """
    Detects hand landmarks in a static image and formats them for gestualis.
    
    Returns:
        A list of dictionaries. Each dictionary represents a hand, mapping 
        landmark indices (0-20) to a numpy array of [x, y, z].
    """
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,  # Strictly for static images
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(image_path)
        results = landmarker.detect(mp_image)
        
        formatted_hands = []
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Convert to numpy array first
                landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
                # Convert to dictionary {0: [x,y,z], 1: [x,y,z], ...} for gestualis
                dict_landmarks = {i: landmarks_np[i] for i in range(21)}
                formatted_hands.append(dict_landmarks)
                
        return formatted_hands

P2 = {2: 3, 4: 10, 6:9, 7:10, 8:10, 10:7}
if __name__ == "__main__":
    G = "1"
    P = "2"

    ######################################
    ######### Storing of file ############
    ######################################

    image_file = f"./kinect_leap_dataset/acquisitions/P{P}/G{G}/10_rgb.png"
    detected_hands = get_hand_landmarks_mediapipe(image_file)

    if detected_hands:
        landmarks = detected_hands[0]

        hand = gestualis.compute.simple_hand(landmarks)
        gestualis.store.store(hand, name=f"{G}.dat")

        print(f"Success: Hand data saved to {G}.dat")
    else:
        print(f"Failed: No hands detected in {image_file}")

    ######################################
    ######### Compare of file ############
    ######################################
    counter = 0
    correct = 0

    for i in range(1, 11):
        image_file = f"./kinect_leap_dataset/acquisitions/P{P}/G{G}/{i}_rgb.png"
        detected_hands = get_hand_landmarks_mediapipe(image_file)
        if detected_hands:
            print("-" * 80)
            for j in detected_hands:
                hand = gestualis.compute.simple_hand(j)
                output = gestualis.store.compare(hand)
                counter += 1
                if output == str(G):
                    correct += 1

                print(output)
    print(f"For Gesture{G}: {correct}/{counter} correct")
