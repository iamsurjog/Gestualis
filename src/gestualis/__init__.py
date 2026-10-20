import pickle
import os
import numpy as np
import numpy.typing as npt
from typing import Callable
from . import compute

# def main() -> None:
#     print("No executable here")

class Gestualis():
    """
        Gestualis object to handle settings
    """

    def __init__(self, rotate: bool = False, fingers: bool = False, path: str = "./signData") -> None:
        self.rotate: bool = rotate
        self.fingers: bool = fingers
        self.path: str = path

    def simple_hand(self, landmarks: np.ndarray):
        """
            Returns a standardized "Gestualis-hand" object from the landmarks generated
        """
        hand = {i: landmarks[i] for i in [0, 1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]}
        if self.rotate:
            quat, theta = compute.calculate_required_rotation(wrist=landmarks[0], p_axis=landmarks[9], p_rotate=landmarks[8])
            for i in hand:
                hand[i] = compute.apply_rotation(point=landmarks[i], quaternion=quat, origin=landmarks[0])

        
        open_fingers = compute.get_fingers(hand)
        vec_thumb = (hand[4] - hand[1])
        vec_index = (hand[8] - hand[5])
        vec_middle = (hand[12] - hand[9])
        vec_ring = (hand[16] - hand[13])
        vec_little = (hand[20] - hand[17])

        return {
                "active": open_fingers,
                "vectors": [vec_index, vec_middle, vec_ring, vec_little, vec_thumb]
                }

    def store(self, hand, name):
        """
            stores the landmark data in a standardized format in a file
        """

        if self.rotate:
            with open(self.path + name + ".hr", 'wb') as f:
                pickle.dump(hand, f)
        else:
            with open(self.path + name + ".h", 'wb') as f:
                pickle.dump(hand, f)
    
    def calculate_angle(self, vec1: npt.NDArray[np.float64], vec2: npt.NDArray[np.float64]) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors not of same length")
        dot = np.dot(vec1, vec2)
        return np.clip(dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0)



    def hand_comparator(self, hand1: list[npt.NDArray[np.float64]],
                      hand2: list[npt.NDArray[np.float64]],
                      innerOperator: Callable[[float], float] = lambda x: x,
                      outerOperator: Callable[[list[float]], float] = sum,
                      finalOperator: Callable[[float], float] = lambda x: x
                      ) -> float:
        angles = []
        for i in range(len(hand1)):
            angle = innerOperator(self.calculate_angle(hand1[i], hand2[i]))
            angles.append(angle)
            
        single_value = outerOperator(angles)
        
        return finalOperator(single_value)

    # TODO: Add finger locking logic
    def compare(self, hand, 
            path="./Data",
            innerOperator: Callable[[float], float] = lambda x: x,
            outerOperator: Callable[[list[float]], float] = sum,
            finalOperator: Callable[[float], float] = lambda x: x
            ):
        s = os.listdir(path)
        d = {}
        file_type = 'hr' if self.rotate else 'h'
        for i in s:
            if i.split('.')[-1] == file_type:
                with open(path + '/' + i, 'rb') as file:
                    temp = pickle.load(file)
                    # print(f"{temp = }")
                    # TODO: Add finger locking
                    if self.fingers:
                        if hand["active"] == temp["active"]:
                            d[self.hand_comparator(hand, temp, innerOperator=innerOperator, outerOperator=outerOperator, finalOperator=finalOperator)] = i
        return d[max( d.keys() )]


