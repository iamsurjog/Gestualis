import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# TODO: 1. Implement getting landmarks and showing format and return
def get_landmarks(img):
    HandLandmarker = mp.tasks.vision.HandLandmarker
    options = HandLandmarkerOptions(
    # HACK: replace this path later on
    base_options=BaseOptions(model_asset_path='/home/randomguy/Downloads/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)
    with HandLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        hand_landmarker_result = landmarker.detect(mp_image)
        if not hand_landmarker_result.hand_landmarks:
            return {} # Return empty dict if no hands found
            
        # 2. Get the landmarks for the first detected hand
        first_hand_landmarks = hand_landmarker_result.hand_landmarks[0]
        
        # 3. Format into {0: np.array([x,y,z]), 1: np.array([x,y,z]), ...}
        formatted_landmarks = {
            i: np.array([landmark.x, landmark.y, landmark.z])
            for i, landmark in enumerate(first_hand_landmarks)
        }
        
        return formatted_landmarks, hand_landmarker_result



