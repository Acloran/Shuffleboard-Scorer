from doctest import OutputChecker
import numpy as np 
import cv2
import keyboard

cam = cv2.imread('images/c1.png')

def callback(x):
    global B_low,B_high,G_low,G_high,R_low,R_high
	#assign trackbar position value to H,S,V High and low variable
    B_low = cv2.getTrackbarPos('low B','controls')
    B_high = cv2.getTrackbarPos('high B','controls')
    G_low = cv2.getTrackbarPos('low G','controls')
    G_high = cv2.getTrackbarPos('high G','controls')
    R_low = cv2.getTrackbarPos('low R','controls')
    R_high = cv2.getTrackbarPos('high R','controls')




cv2.namedWindow('controls',2)
cv2.resizeWindow("controls", 550,10);

B_low = 0
B_high = 255
G_low= 0
G_high = 255
R_low= 0
R_high = 255


#create trackbars for high,low H,S,V 
cv2.createTrackbar('low B','controls',0,255,callback)
cv2.createTrackbar('high B','controls',255,255,callback)

cv2.createTrackbar('low G','controls',0,255,callback)
cv2.createTrackbar('high G','controls',255,255,callback)

cv2.createTrackbar('low R','controls',0,255,callback)
cv2.createTrackbar('high R','controls',255,255,callback)



while True:

    # og = cam
    
    # pts1 = np.float32([[297,317],[734,319],[732,541],[292,528]])
    # pts2 = np.float32([[0,0],[600,0],[600,300],[0,300]])

    # M = cv2.getPerspectiveTransform(pts1,pts2)

    # dst = cv2.warpPerspective(og,M,(620,300))
    dst = cam
    cv2.imshow('raw',dst) 
    #recolor Image
    frame = cv2.bitwise_not(dst)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    blue_bgr_low = np.array([B_low, G_low, R_low], np.uint8)
    blue_bgr_high = np.array([B_high, G_high, R_high], np.uint8)

	
    blueMask = cv2.inRange(frame, blue_bgr_low, blue_bgr_high)

    cv2.imshow('masked img', blueMask)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()