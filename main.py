import numpy as np 
import cv2
import keyboard

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    cv2.circle(frame,(400,400),8,(255,0,0),2)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()