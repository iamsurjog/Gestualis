import modules
import numpy as np

def main(landmarks):
    quat, theta = modules.calculate_flattening_rotation(wrist=landmarks[0], p_axis=landmarks[9], p_rotate=landmarks[8])
    rotated = {}
    rotated[9] = landmarks[9]
    rotated[0] = landmarks[0]
    for i in [1, 4, 5, 8, 12, 13, 16, 17, 20]:
        rotated[i] = modules.apply_rotation(landmark=landmarks[i], quaternion=quat, wrist_pos=landmarks[0])
    
    thumb = np.linalg.norm(rotated[4] - rotated[1])
    index = np.linalg.norm(rotated[8] - rotated[5])
    middle = np.linalg.norm(rotated[12] - rotated[9])
    ring = np.linalg.norm(rotated[16] - rotated[13])
    little = np.linalg.norm(rotated[20] - rotated[17])
    return thumb, index, middle, ring, little
    
