from doctest import OutputChecker
import numpy as np 
import cv2
import keyboard

cam = cv2.VideoCapture(0)#,cv2.CAP_DSHOW)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensionns to cam object (not cap)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

def callback(x):
    global H_low,H_high,S_low,S_high,V_low,V_high
	#assign trackbar position value to H,S,V High and low variable
    H_low = cv2.getTrackbarPos('low H','controls')
    H_high = cv2.getTrackbarPos('high H','controls')
    S_low = cv2.getTrackbarPos('low S','controls')
    S_high = cv2.getTrackbarPos('high S','controls')
    V_low = cv2.getTrackbarPos('low V','controls')
    V_high = cv2.getTrackbarPos('high V','controls')




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

    ret, og = cam.read()
    
    pts1 = np.float32([[297,317],[734,319],[732,541],[292,528]])
    pts2 = np.float32([[0,0],[600,0],[600,300],[0,300]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(og,M,(620,300))
    
    #recolor Image
    frame = cv2.bitwise_not(dst)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('raw',dst)
    cv2.imshow('hsv',hsv) 
    cv2.imshow('inverted',frame)
    blue_bgr_low = np.array([H_low, S_low, V_low], np.uint8)
    blue_bgr_high = np.array([H_high, S_high, V_high], np.uint8)

	
    mask = cv2.inRange(hsv, blue_bgr_low, blue_bgr_high)


    median = cv2.medianBlur(mask,7)

    kernel = np.ones((5,5),np.uint8)
    kernel2 = np.ones((3,3),np.uint8)

    closed = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel)
    eroded = cv2.erode(closed,kernel2,iterations = 1)

    


    cv2.imshow('masked img', eroded)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()