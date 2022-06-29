import numpy as np 
import cv2
import keyboard

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()