import math
import cv2
import mediapipe as mp
import time


class handTracker():
    def __init__(self, mode=False, numhands = 2, detectionConf = 0.5, trackConf = 0.5):
        self.mode = mode
        self.numhands = numhands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands

        ##Creating Hands with Parameters
        self.hands = self.mpHands.Hands(self.mode, self.numhands, self.detectionConf, self.trackConf)

        ##Getting Drawings from MediaPipe
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHand(self, image, draw=True):
        ##Convert to RGB
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.result = self.hands.process(rgbImage)

        if self.result.multi_hand_landmarks:
            for landmarks in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, landmarks, self.mpHands.HAND_CONNECTIONS)
        return image


    def findPosition(self, image, handNum = 0, draw=True):
        xList = []
        yList = []
        box = []

        self.landmarkList = []

        if self.result.multi_hand_landmarks:
            curHand = self.result.multi_hand_landmarks[handNum]

            for index, lm in enumerate(curHand.landmark):
                #each point corresponds to an index in MediaPipe
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                ##Add to list
                self.landmarkList.append([index, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 5, (255, 255, 0), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            box = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(image, (xmin-20, ymin-20), (xmax + 20, ymax + 20), (0,255,0), 2)

        return self.landmarkList, box

    def fingersUp(self):
        fingers = []

        #For Thumb
        if self.landmarkList[self.tipIds[0]][1] > self.landmarkList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #Rest of Fingers
        for index in range(1,5):
            if self.landmarkList[self.tipIds[index]][2] > self.landmarkList[self.tipIds[index] - 2][2]:
                fingers.append(0)
            else:
                fingers.append(1)

        return fingers

    def findDistance(self, p1, p2, image, draw=True, r=15, t=3):
        x1, y1 = self.landmarkList[p1][1:]
        x2, y2 = self.landmarkList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(image, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2-y1)
        return length, image, [x1, y1, x2, y2, cx, cy]


def main():
    capture = cv2.VideoCapture(0)
    prevTime = 0
    curTime = 0

    tracker = handTracker()

    while True:
        success, image = capture.read()
        image = tracker.findHand(image)
        landmarkList = tracker.findPosition(image)

        if len(landmarkList) != 0:
            print(landmarkList[0])



        ##fps counter
        curTime = time.time()
        fps = 1 / (curTime - prevTime)
        prevTime = curTime

        cv2.putText(image, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1)
        cv2.imshow("Image", image)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()