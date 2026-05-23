import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import HandDetectorModule as hdm
import autopy

smoothening = 3

wcam,hcam = 640,480
fr = 100
wscrn,hscrn = autopy.screen.size()

cap = cv.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)

detector = hdm.handDetector()

plocx,plocy = 0,0
clocx,clocy = 0,0

while True:
    bool,img = cap.read()
    if not bool:
        break

    img = cv.flip(img,1)

    img = detector.findHands(img)
    lmlist = detector.findposition(img,False)

    cv.rectangle(img,(fr-50,fr-50),(wcam-fr+50,hcam-fr-30),(255,0,255),4)

    if len(lmlist) != 0:
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]
        fingers = detector.fingerUp()
        # print(fingers)
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1,(fr-50,wcam-fr+50),(0,wscrn))
            y3 = np.interp(y1,(fr-50,hcam-fr-30),(0,hscrn))

            clocx = plocx + (x3-plocx)/smoothening
            clocy = plocy + (y3-plocy)/smoothening

            autopy.mouse.move(clocx,clocy)
            cv.circle(img,(x1,y1),8,(0,255,0),cv.FILLED)

        elif fingers[1] == 1 and fingers[2] == 1:
            d = detector.findDistance(8,12)
            if d<24:
                autopy.mouse.click()

        plocx,plocy = clocx,clocy


    cv.imshow("Image",img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()