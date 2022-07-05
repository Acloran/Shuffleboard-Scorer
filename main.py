import numpy as np 
import cv2
import keyboard

#cam = cv2.VideoCapture(0)#,cv2.CAP_DSHOW)

#cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensionns to cam object (not cap)
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

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
    #ret, og = cam.read()
    og = cam
    frame = cv2.bitwise_not(og)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    

    bgr_low = np.array([B_low, G_low, R_low], np.uint8)
    bgr_high = np.array([B_high, G_high, R_high], np.uint8)

	#making mask for hsv range
    mask = cv2.inRange(frame, bgr_low, bgr_high)
    median = cv2.medianBlur(mask,5)
    #print (mask)
    res = cv2.bitwise_and(cam, cam, mask=mask)

    cv2.imshow('blurred',median)
    cv2.imshow('res',res)

    #cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break
#cam.release()
cv2.destroyAllWindows()