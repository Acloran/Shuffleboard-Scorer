import numpy as np 
import cv2
import keyboard

cam = cv2.VideoCapture(0)

while True:
    cam.read()
    cv2.imshow('Camera', cam)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break