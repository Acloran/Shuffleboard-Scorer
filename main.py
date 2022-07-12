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




cv2.namedWindow('controls',2)
cv2.resizeWindow("controls", 550,10)

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

def drawRedorBlueCircle(imgin, imgout, x, y):
    colormask = np.zeros((300,620,1), np.uint8)
    cv2.circle(colormask,center,12,(255),12)
    offset = 18
    xVal = int(x)
    yVal = int(y)
    # redTotal = 0
    # redTotal += imgin.item(xVal,yVal+10,2)
    # redTotal += imgin.item(xVal,yVal-10,2)
    # redTotal += imgin.item(xVal+10,yVal,2)
    # redTotal += imgin.item(xVal-10,yVal,2)
    # blueVal = redTotal/4

    #puckBox = imgin[(xVal-offset):(xVal+offset), (yVal-offset):(yVal+offset)]
    #cv2.rectangle(imgin, (xVal-offset, yVal-offset), (xVal+offset, yVal+offset), (0,255,0), 1)
    #cv2.imshow('window',imgin)
    #b = puckBox[:,:,2]
    
    #blueVal = np.mean(b, axis=None)
    #print(blueVal)

    blueVal = 230

    if blueVal < 230:
        circleColor = (255,0,0)
    else:
        circleColor = (0,0,255)

    #img2gray = cv2.cvtColor(imgin,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(colormask, 10, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(imgin, imgin, mask=mask)
    cv2.imshow('reaultofmask', result)
    #circleColor = (255,255,255)
    blueVal = round(blueVal, 1)
    strBlueVal = str(blueVal)
    
    cv2.putText(imgout, strBlueVal, (xVal+22,yVal+5), cv2.FONT_HERSHEY_SIMPLEX, .5, circleColor, 2, cv2.LINE_AA)
    cv2.circle(imgout,center,18,circleColor,2)
    # draw the center of the circle
    cv2.circle(imgout,center,6,circleColor,-1)
    

def findAndDrawCircles(img, bgr_low, bgr_high):
	
    mask = cv2.inRange(img, bgr_low, bgr_high)
    median = cv2.medianBlur(mask,7)

    kernel = np.ones((5,5),np.uint8)
    kernel2 = np.ones((3,3),np.uint8)

    closed = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel)
    eroded = cv2.erode(closed,kernel2,iterations = 1)

    ret,thresh = cv2.threshold(eroded,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE) 

    cv2.imshow('blue',closing)  
    for i in contours[:]:
        (x,y),radius = cv2.minEnclosingCircle(i)
        center = (int(x),int(y))
        radius = int(radius)
        #cv2.circle(img,center,radius,(0,255,0),2)
        if radius>10 and radius<25:
            cv2.circle(outputImg,center,18,(255,0,0),2)
            # draw the center of the circle
            cv2.circle(outputImg,center,6,(255,0,0),-1)

while True:

    #define an all black image
    outputImg = np.zeros((300,620,3), np.uint8)
    #draw scoring lines and numbers
    cv2.line(outputImg,(392,0),(392,300),(0,255,0),2)
    cv2.line(outputImg,(498,0),(498,300),(0,255,0),2)
    cv2.line(outputImg,(600,0),(600,300),(0,255,0),2)
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(outputImg,'1',(335,165), font, 2.2,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(outputImg,'2',(426,165), font, 2.2,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(outputImg,'3',(525,165), font, 2.2,(0,255,0),2,cv2.LINE_AA)


    ret, og = cam.read()
    
    #straighten image
    #og = cam
    
    pts1 = np.float32([[297,317],[734,319],[732,541],[292,528]])
    pts2 = np.float32([[0,0],[600,0],[600,300],[0,300]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(og,M,(620,300))
    cv2.imshow('raw',dst) 
    #recolor Image
    frame = cv2.bitwise_not(dst)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #Making the Mask for Blue Pucks
    bgr_low = np.array([B_low, G_low, R_low], np.uint8)
    bgr_high = np.array([B_high, G_high, R_high], np.uint8)

    blue_bgr_low = np.array([30, 0, 59], np.uint8)
    blue_bgr_high = np.array([210, 135, 255], np.uint8)

	
    blueMask = cv2.inRange(frame, blue_bgr_low, blue_bgr_high)
    blueMask = cv2.medianBlur(blueMask,7)

    kernel = np.ones((7,7),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)

    closing = cv2.morphologyEx(blueMask, cv2.MORPH_CLOSE, kernel)
    #closing = cv2.erode(closing,kernel2,iterations = 1)
    cv2.imshow('blue',closing)  
    # #Drawing detected circles

    # v = np.mean(closing)
    # print(v)
    # param1 = int(min(255, (1.0+.6) * v))

    # bluecircles = cv2.HoughCircles(closing,cv2.HOUGH_GRADIENT,1.2,12,
    #                         param1=30,param2=20,minRadius=12,maxRadius=25)
    # if bluecircles is not None:
    #     bluecircles = np.uint16(np.around(bluecircles))
    # #print(circles)

    #     for i in bluecircles[0,:]:
    #         # draw the outer circle
    #         cv2.circle(outputImg,(i[0],i[1]),18,(255,0,0),2)
    #         # draw the center of the circle
    #         cv2.circle(outputImg,(i[0],i[1]),2,(255,0,0),3)
    
    

    ret,thresh = cv2.threshold(closing,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE) 

    cv2.imshow('blue',closing)  
    for i in contours[:]:
        (x,y),radius = cv2.minEnclosingCircle(i)
        center = (int(x),int(y))
        radius = int(radius)
        #cv2.circle(img,center,radius,(0,255,0),2)
        if radius>4 and radius<15:
            drawRedorBlueCircle(dst, outputImg, x, y)
            #cv2.circle(outputImg,center,18,(255,0,0),2)
            # draw the center of the circle
            #cv2.circle(outputImg,center,6,(255,0,0),-1)

    #making the red mask

    # red_bgr_low = np.array([0, 166, 0], np.uint8)
    # red_bgr_high = np.array([255, 255, 102], np.uint8)

	
    # redMask = cv2.inRange(frame, red_bgr_low, red_bgr_high)
    # redMask = cv2.medianBlur(redMask,5)
    # redMask = cv2.morphologyEx(redMask, cv2.MORPH_CLOSE, kernel)
    # redMask = cv2.erode(redMask,kernel2,iterations = 1)

    # #Drawing detected circles
    # redcircles = cv2.HoughCircles(redMask,cv2.HOUGH_GRADIENT,1,10,
    #                         param1=50,param2=20,minRadius=4,maxRadius=0)
    # if redcircles is not None:
    #     redcircles = np.uint16(np.around(redcircles))
    #     #print(redcircles)
    #     for i in redcircles[0,:]:
    #         # draw the outer circle
    #         cv2.circle(outputImg,(i[0],i[1]),18,(0,0,255),2)
    #         # draw the center of the circle
    #         cv2.circle(outputImg,(i[0],i[1]),2,(0,0,255),3)
    # cv2.imshow('red',redMask)




    #res = cv2.bitwise_and(dst, dst, mask=blueMask)

    

    # cv2.imshow('blurred',mask)
    cv2.imshow('result',outputImg)

    #cv2.imshow('Camera', frame)

    if cv2.waitKey(400) == ord('q'):
        while True:
            if cv2.waitKey(1) == ord('x'):
                break
        break
cam.release()
cv2.destroyAllWindows()