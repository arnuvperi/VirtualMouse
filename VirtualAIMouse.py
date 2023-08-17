import autopy.screen
import cv2
import numpy as np
import HandTracking as ht
import time

capture = cv2.VideoCapture(0)

##width and height
wCam, hCam = 1280, 720
wScr, hScr = autopy.screen.size()
frameR = 200
frameRH = 0
frameRW = 200

smooth = 7
prevX, prevY = 0, 0
curX, curY = 0, 0

capture.set(3, wCam)
capture.set(4, hCam)
prevTime = 0


tracker = ht.handTracker(numhands = 1)

while True:
    success, image = capture.read()

    # Find hand Landmarks

    image = tracker.findHand(image)
    landmarkL, box = tracker.findPosition(image)

    # Tip of index and middle

    if(len(landmarkL) != 0):
        #index
        x1, y1 = landmarkL[8][1:]
        #middle
        x2, y2 = landmarkL[12][1:]

        # Check finger up
        fingers = tracker.fingersUp()
        cv2.rectangle(image, (frameRW, frameRH), (wCam - frameRW, hCam - frameR), (255, 255, 0), 1)

        # If only index, moving
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert Coord


            x3 = np.interp(x1, (frameRW, wCam - frameRW), (0, wScr))
            y3 = np.interp(y1, (frameRH, hCam - frameR), (0, hScr))


            # Smoothen
            curX = prevX + (x3 - prevX) / smooth
            curY = prevY + (y3 - prevY) / smooth


            # Moving
            autopy.mouse.move(wScr-curX,curY)
            cv2.circle(image, (x1, y1), 10, (255,255,0), cv2.FILLED)

        # If not, click
        if fingers[1] == 1 and fingers[2] == 1:
            length, image, lineInfo = tracker.findDistance(8,12, image)
            if length < 60:
                cv2.circle(image, (lineInfo[4], lineInfo[5]), 10, (0,255,0), cv2.FILLED)
                autopy.mouse.click()

        prevX, prevY = curX, curY


    ##fps counter
    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime
    cv2.putText(image, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1)

    #Display
    cv2.imshow("Image", image)
    cv2.waitKey(1)