import pickle
import os
from . import compute
from . import camera

def get_points(img):
    pts, raw = camera.get_landmarks(img)
    return compute.simple_hand(pts), pts, raw


def compare(hand, path="./Data"):
    s = os.listdir(path)
    d = {}
    for i in s:
        with open(path + '/' + i) as file:
            temp = pickle.load(path + '/' + i)
            # TODO: 2. add the functional transforms
            d[compute.hand_comparator(hand, temp)] = i
    return d[max( d.keys() )][:-4]
            


def store(data, path="./Data", name="temp.dat"):
    with open(path + "/" + name, "wb") as file:
        pickle.dump(data, file)


def main() -> None:
    print("No executable here")
