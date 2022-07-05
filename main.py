from doctest import OutputChecker
import numpy as np 
import cv2
import keyboard

cam = cv2.VideoCapture(0)#,cv2.CAP_DSHOW)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensionns to cam object (not cap)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

#cam = cv2.imread('images/c1.png')

def callback(x):
    global B_low,B_high,G_low,G_high,R_low,R_high
	#assign trackbar position value to H,S,V High and low variable
    B_low = cv2.getTrackbarPos('low B','controls')
    B_high = cv2.getTrackbarPos('high B','controls')
    G_low = cv2.getTrackbarPos('low G','controls')
    G_high = cv2.getTrackbarPos('high G','controls')
    R_low = cv2.getTrackbarPos('low R','controls')
    R_high = cv2.getTrackbarPos('high R','controls')




# cv2.namedWindow('controls',2)
# cv2.resizeWindow("controls", 550,10);

# B_low = 0
# B_high = 255
# G_low= 0
# G_high = 255
# R_low= 0
# R_high = 255


# #create trackbars for high,low H,S,V 
# cv2.createTrackbar('low B','controls',0,255,callback)
# cv2.createTrackbar('high B','controls',255,255,callback)

# cv2.createTrackbar('low G','controls',0,255,callback)
# cv2.createTrackbar('high G','controls',255,255,callback)

# cv2.createTrackbar('low R','controls',0,255,callback)
# cv2.createTrackbar('high R','controls',255,255,callback)



while True:

    #define an all black image
    #outputImg = np.zeros((300,620,3), np.uint8)

    ret, og = cam.read()
    
    #straighten image
    #og = cam
    
    pts1 = np.float32([[297,317],[734,319],[732,541],[292,528]])
    pts2 = np.float32([[0,0],[600,0],[600,300],[0,300]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(og,M,(620,300))
    outputImg = dst
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
    bluecircles = cv2.HoughCircles(blueMask,cv2.HOUGH_GRADIENT,1,8,
                            param1=50,param2=20,minRadius=4,maxRadius=0)
    if bluecircles is not None:
        bluecircles = np.uint16(np.around(bluecircles))
    #print(circles)

        for i in bluecircles[0,:]:
            # draw the outer circle
            cv2.circle(outputImg,(i[0],i[1]),18,(255,0,0),2)
            # draw the center of the circle
            cv2.circle(outputImg,(i[0],i[1]),2,(255,0,0),3)
    

    #making the red mask

    red_bgr_low = np.array([0, 162, 0], np.uint8)
    red_bgr_high = np.array([255, 255, 76], np.uint8)

	
    redMask = cv2.inRange(frame, red_bgr_low, red_bgr_high)
    redMask = cv2.medianBlur(redMask,5)

    #Drawing detected circles
    redcircles = cv2.HoughCircles(redMask,cv2.HOUGH_GRADIENT,1,8,
                            param1=50,param2=20,minRadius=4,maxRadius=0)
    if redcircles is not None:
        redcircles = np.uint16(np.around(redcircles))
        #print(redcircles)
        for i in redcircles[0,:]:
            # draw the outer circle
            cv2.circle(outputImg,(i[0],i[1]),18,(0,0,255),2)
            # draw the center of the circle
            cv2.circle(outputImg,(i[0],i[1]),2,(0,0,255),3)
   




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