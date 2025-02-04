import os
import time
import pickle
import mediapipe as mp
import cv2


class HandDetector:
    def __init__(self):
        mpHands = mp.solutions.hands
        self.hand = mpHands.Hands()
        self.cap = cv2.VideoCapture(0)

    def capture(self):
        """
        captures the 10 frames and processes it
        :return: the position of all landmarks in the 10 frames in form of [[[x, y, z]<- each landmark, [x, y, z]...]<- each frame, [[x, y, z], [x, y, z]...] ...]
        """
        res = []
        for i in range(10):
            success, img = self.cap.read()
            res.append(img)
            time.sleep(0.1)
            img = cv2.flip(img, 1)
            # for the detection
            res[i] = cv2.flip(res[i], 1)
            res[i] = cv2.cvtColor(res[i], cv2.COLOR_BGR2RGB)
            res[i] = self.hand.process(res[i])
            res[i] = res[i].multi_hand_landmarks
            if res[i] is None:
                raise RuntimeError("Hand either not detected or complete hand not on screen.")
            temp = []
            for j in res[i]:
                for k in j.landmark:
                    temp.append([k.x, k.y, k.z])
                    x,y,_ = img.shape
                    cv2.circle(img, (int(k.x*y), int(k.y*x)),  5, (255, 0, 255), 5)
            res[i] = temp
            cv2.imshow("HandDetector", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.waitKey()
        return res

    def detect(self, res):
        """
        Takes in 10 images from direct openCV format and returns the formatted result
        :param res: list of 10 images from openCV
        :return: the position of all landmarks in the 10 frames in form of [[[x, y, z]<- each landmark, [x, y, z]...]<- each frame, [[x, y, z], [x, y, z]...] ...]
        """
        for i in range(len(res)):
            res[i] = cv2.flip(res[i], 1)
            res[i] = cv2.cvtColor(res[i], cv2.COLOR_BGR2RGB)
            res[i] = self.hand.process(res[i])
            res[i] = res[i].multi_hand_landmarks
            temp = []
            for j in res[i]:
                for k in j.landmark:
                    temp.append([k.x, k.y, k.z])
            res[i] = temp
        return res


    def distance(self, a, b):
        """
        Finds the cosec value between two set of landmark values given.
        :param a: the position of all landmarks in the 10 frames in form of [[[x, y, z]<- each landmark, [x, y, z]...]<- each frame, [[x, y, z], [x, y, z]...] ...]
        :param b: the position of all landmarks in the 10 frames in form of [[[x, y, z]<- each landmark, [x, y, z]...]<- each frame, [[x, y, z], [x, y, z]...] ...]
        :return: cosec distance between vector a and vector b
        """
        dot_prod = 0
        moda = 0
        modb = 0
        for i in range(min(len(b), len(a))):
            for j in range(min(len(b[i]), len(a[i]))):
                dot_prod += a[i][j][0]*b[i][j][0]
                dot_prod += a[i][j][1]*b[i][j][1]
                dot_prod += a[i][j][2]*b[i][j][2]
                moda += a[i][j][0]**2 + a[i][j][1]**2 + a[i][j][2]**2
                modb += b[i][j][0]**2 + b[i][j][1]**2 + b[i][j][2]**2
        dot_prod /= (moda**(0.5))*(modb**(0.5))
        cosec = 1/((1-dot_prod**2)**(0.5))
        return cosec

    def store(self, res, label, path="./signData"):
        """
        Stores the landmark positions in a .dat file in given location.
        :param res: the position of all landmarks in the 10 frames in form of [[[x, y, z]<- each landmark, [x, y, z]...]<- each frame, [[x, y, z], [x, y, z]...] ...]
        :param label: Name of the sign
        :param path: location of all the .dat file
        :return:
        """
        if not os.path.exists(path):
            os.makedirs(path)
        if ".dat" in label:
            f = open(path + "/" + label, "wb")
        else:
            f = open(path + "/" + label + ".dat", "wb")
        pickle.dump(res, f)
        f.close()

    def check(self, res, path="/signData"):
        try:
            l = os.listdir("." + path)
        except FileNotFoundError:
            raise FileNotFoundError("Path not found")
        dats = [i for i in l if ".dat" in i]
        if len(dats) == 0:
            raise FileNotFoundError("No data found or wrong path")
        vals = dict()
        for i in dats:
            f = open(path + "/" + i, "rb")
            dat = pickle.load(f)
            vals[self.distance(res, dat)] = i
        m = max(vals.keys())
        ans = vals[m]
        return "".join(ans.split(".")[:-1])






