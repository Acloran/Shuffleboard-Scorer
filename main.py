import numpy as np 
import cv2
import keyboard

cam = cv2.VideoCapture(0)#,cv2.CAP_DSHOW)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensionns to cam object (not cap)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

while True:
    ret, frame = cam.read()

    cv2.circle(frame,(640,512),10,(255,0,0),2)
    cv2.circle(frame,(600,512),10,(0,0,255),2)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()