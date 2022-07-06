from doctest import OutputChecker
import numpy as np 
import cv2
import keyboard

backgroundimg = cv2.imread('images/warpedbackground.png')

cam = cv2.VideoCapture(0)#,cv2.CAP_DSHOW)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensionns to cam object (not cap)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

fgbg1 = cv2.createBackgroundSubtractorMOG2(); 



while True:

    ret, og = cam.read()
    
    
    pts1 = np.float32([[297,317],[734,319],[732,541],[292,528]])
    pts2 = np.float32([[0,0],[600,0],[600,300],[0,300]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(og,M,(620,300))

    difference = cv2.subtract(dst,backgroundimg)

    grayDifference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    bgr_low = np.array([0, 0, 107], np.uint8)
    bgr_high = np.array([175, 255, 255], np.uint8)

    final = cv2.inRange(grayDifference, 1, 255)

    #fgmask1 = fgbg1.apply(dst)

    # cv2.imshow('raw',dst) 
    # #recolor Image
    # frame = cv2.bitwise_not(dst)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # blue_bgr_low = np.array([B_low, G_low, R_low], np.uint8)
    # blue_bgr_high = np.array([B_high, G_high, R_high], np.uint8)

	
    #blueMask = cv2.inRange(frame, blue_bgr_low, blue_bgr_high)

    cv2.imshow('masked img', result)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()