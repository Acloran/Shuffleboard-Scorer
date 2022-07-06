import numpy as np 
import cv2
import keyboard

cam = cv2.VideoCapture(0)#,cv2.CAP_DSHOW)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensionns to cam object (not cap)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

ret,frame = cam.read()

while(True):
    cv2.imshow('img1',frame) #display the captured image
    pts1 = np.float32([[297,317],[734,319],[732,541],[292,528]])
    pts2 = np.float32([[0,0],[600,0],[600,300],[0,300]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(frame,M,(620,300))
    if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
        cv2.imwrite('images/warpedbackground.png',dst)
        cv2.destroyAllWindows()
        break

cam.release()