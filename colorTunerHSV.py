from doctest import OutputChecker
import numpy as np 
import cv2
import keyboard

cam = cv2.imread('images/c1.png')

def callback(x):
    global H_low,H_high,S_low,S_high,V_low,V_high
	#assign trackbar position value to H,S,V High and low variable
    B_low = cv2.getTrackbarPos('low H','controls')
    B_high = cv2.getTrackbarPos('high H','controls')
    G_low = cv2.getTrackbarPos('low S','controls')
    G_high = cv2.getTrackbarPos('high S','controls')
    R_low = cv2.getTrackbarPos('low V','controls')
    R_high = cv2.getTrackbarPos('high V','controls')




cv2.namedWindow('controls',2)
cv2.resizeWindow("controls", 550,10);

H_low = 0
H_high = 179
S_low= 0
S_high = 255
V_low= 0
V_high = 255


#create trackbars for high,low H,S,V 
cv2.createTrackbar('low H','controls',0,179,callback)
cv2.createTrackbar('high H','controls',179,179,callback)

cv2.createTrackbar('low S','controls',0,255,callback)
cv2.createTrackbar('high S','controls',255,255,callback)

cv2.createTrackbar('low V','controls',0,255,callback)
cv2.createTrackbar('high V','controls',255,255,callback)



while True:

    og = cam
    
    pts1 = np.float32([[297,317],[734,319],[732,541],[292,528]])
    pts2 = np.float32([[0,0],[600,0],[600,300],[0,300]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(og,M,(620,300))
    
    #recolor Image
    frame = cv2.bitwise_not(dst)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv',hsv) 
    blue_bgr_low = np.array([H_low, S_low, V_low], np.uint8)
    blue_bgr_high = np.array([H_high, S_high, V_high], np.uint8)

	
    blueMask = cv2.inRange(hsv, blue_bgr_low, blue_bgr_high)

    cv2.imshow('masked img', blueMask)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()