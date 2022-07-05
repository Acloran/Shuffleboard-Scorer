import numpy as np 
import cv2
import keyboard

cam = cv2.VideoCapture(0)#,cv2.CAP_DSHOW)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensionns to cam object (not cap)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

ret,frame = cam.read()

while(True):
    cv2.imshow('img1',frame) #display the captured image
    if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
        cv2.imwrite('images/c1.png',frame)
        cv2.destroyAllWindows()
        break

cam.release()