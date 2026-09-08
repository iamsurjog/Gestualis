from typing import Callable
import numpy as np
import numpy.typing as npt
import pickle
import os

def calculate_angle(vec1: npt.NDArray[np.float64], vec2: npt.NDArray[np.float64]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors not of same length")
    dot = np.dot(vec1, vec2)
    return np.clip(dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0)


def hand_comparator(hand1: list[npt.NDArray[np.float64]],
                  hand2: list[npt.NDArray[np.float64]],
                  innerOperator: Callable[[float], float] = lambda x: x,
                  outerOperator: Callable[[list[float]], float] = sum,
                  finalOperator: Callable[[float], float] = lambda x: x
                  ) -> float:
    angles = []
    for i in range(len(hand1)):
        angle = innerOperator(calculate_angle(hand1[i], hand2[i]))
        angles.append(angle)
        
    single_value = outerOperator(angles)
    
    return finalOperator(single_value)


def compare(hand, 
            path="./Data",
            innerOperator: Callable[[float], float] = lambda x: x,
            outerOperator: Callable[[list[float]], float] = sum,
            finalOperator: Callable[[float], float] = lambda x: x
            ):
    s = os.listdir(path)
    d = {}
    for i in s:
        with open(path + '/' + i, 'rb') as file:
            temp = pickle.load(file)
            # print(f"{temp = }")
            d[hand_comparator(hand, temp, innerOperator=innerOperator, outerOperator=outerOperator, finalOperator=finalOperator)] = i
    return d[max( d.keys() )][:-4]
            


def store(data, path="./Data", name="temp.dat"):
    with open(path + "/" + name, "wb") as file:
        pickle.dump(data, file)
