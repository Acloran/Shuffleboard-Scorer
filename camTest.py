from doctest import OutputChecker
import numpy as np 
import cv2
import keyboard

cam = cv2.VideoCapture(0)#,cv2.CAP_DSHOW)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensionns to cam object (not cap)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

while True:
    img = cam.read()
    cv2.imshow('result',img)

    #cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        while True:
            if cv2.waitKey(1) == ord('x'):
                break
        break
cam.release()
cv2.destroyAllWindows()