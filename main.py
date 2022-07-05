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

#define an all black image
outputImg = np.zeros((620,300,3), np.uint8)

while True:
    #ret, og = cam.read()
    #straighten image
    og = cam

    pts1 = np.float32([[297,317],[734,319],[732,541],[292,528]])
    pts2 = np.float32([[0,0],[600,0],[600,300],[0,300]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(og,M,(620,300))
    
    #recolor Image
    frame = cv2.bitwise_not(dst)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Making the Mask for Blue Pucks
    #bgr_low = np.array([B_low, G_low, R_low], np.uint8)
    #bgr_high = np.array([B_high, G_high, R_high], np.uint8)

    blue_bgr_low = np.array([0, 156, 137], np.uint8)
    blue_bgr_high = np.array([160, 255, 255], np.uint8)

	
    blueMask = cv2.inRange(frame, blue_bgr_low, blue_bgr_high)
    blueMask = cv2.medianBlur(blueMask,5)

    #Drawing detected circles
    circles = cv2.HoughCircles(blueMask,cv2.HOUGH_GRADIENT,1,8,
                            param1=50,param2=20,minRadius=4,maxRadius=0)

    circles = np.uint16(np.around(circles))
    print(circles)
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(outputImg,(i[0],i[1]),i[2],(255,0,0),-1)
        # draw the center of the circle
        cv2.circle(outputImg,(i[0],i[1]),2,(255,0,0),3)
    

    #res = cv2.bitwise_and(dst, dst, mask=blueMask)

    #draw scoring lines
    cv2.line(outputImg,(392,0),(392,300),(0,255,0),2)
    cv2.line(outputImg,(498,0),(498,300),(0,255,0),2)


    # cv2.imshow('blurred',mask)
    cv2.imshow('result',outputImg)

    #cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break
#cam.release()
cv2.destroyAllWindows()