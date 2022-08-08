from doctest import OutputChecker
import numpy as np 
import cv2
import keyboard
from picamera2 import Picamera2
from main import *
from arducam.focus.Focuser import Focuser

def recolorTableImg(img):
    frame = cv2.bitwise_not(img)

    bgr_low = np.array([130, 110, 63], np.uint8)
    bgr_high = np.array([230, 200, 162], np.uint8)
	
    greenMask = cv2.inRange(frame, bgr_low, bgr_high)
    greenMask = cv2.medianBlur(greenMask,7)

    kernel = np.ones((9,9),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)

    closing = cv2.morphologyEx(greenMask, cv2.MORPH_CLOSE, kernel)
    ret,thresh = cv2.threshold(closing,127,255,0)
    return thresh

def findEdgePoint(img,x1,y1):
    #find the edge of the table
    width = 80
    height = 80
    halfWidth = int(width/2)
    halfHeight = int(height/2)

    tlX = x1-halfWidth
    tlY = y1-halfHeight

    poi = img[tlY:tlY+height,tlX:tlX+width]
    
    processedImg = recolorTableImg(poi)
    
    tl = processedImg.item(0,0)
    tr = processedImg.item(0,width-1)
    bl = processedImg.item(height-1,0)
    br = processedImg.item(height-1,width-1)
    #compares the top left and bottom left pixels to the top right and bottom right pixels to determine if the edge is vertical or horizontal and which direction it is facing
    if (tl + bl)/2 > (tr + br)/2:
        rangeVal1 = width-1
        rangeVal2 = 0
        rangeVal3 = -1
        valX1 = width - 1
        valX2 = width - 1
        valY1 = 0
        valY2 = height-1
        iX = True
        leftWhite = True
        #print('case1')
    elif (tl + bl)/2 < (tr + br)/2:
        rangeVal1 = 0
        rangeVal2 = width-1
        rangeVal3 = 1
        valX1 = 0
        valX2 = 0
        valY1 = 0
        valY2 = height-1
        iX = True
        leftWhite = False
        #print('case2')
    else:
        rangeVal1 = 0
        rangeVal2 = height-1
        rangeVal3 = 1
        valX1 = 0
        valX2 = width-1
        valY1 = 0
        valY2 = 0
        iX = False
        #print('case3')

    #(y,x) = processedImg.shape
    dist1 = halfHeight
    dist2 = halfHeight
    done1 = False
    done2 = False
    for i in range(rangeVal1,rangeVal2,rangeVal3):
        #sets x or y looping
        if iX:
            valX1 = i
            valX2 = i
        else:
            valY1 = i
            valY2 = i
        
        if processedImg.item(valY1,valX1)==255 and not done1:
            dist1 = i
            done1 = True
            
        #print(processedImg.item(valY2,valX2))
        if processedImg.item(valY2,valX2)==255 and not done2:
            dist2 = i
            done2 = True
            

        if done1 and done2:
            break

    average = (dist1+dist2)/2.0
    #print('dist1:',dist1,'dist2:',dist2,)
    #cv2.imshow('processed',processedImg)
    #cv2.imshow('poi',poi)
    if iX:
        outX = tlX + int(average)
        outY = tlY + halfHeight
    else:
        outX = tlX+halfWidth
        outY = tlY+int(average)

    return (outX,outY)

def calculateCorner(img,x1,y1,x2,y2,x3,y3,x4,y4):

    x_1,y_1 = findEdgePoint(img,x1,y1)
    x_2,y_2 = findEdgePoint(img,x2,y2)
    x_3,y_3 = findEdgePoint(img,x3,y3)
    x_4,y_4 = findEdgePoint(img,x4,y4)

    if x_2-x_1 != 0:
        m1 = (y_2-y_1)/(x_2-x_1)
    else:
        m1 = 2147483647
    if x_4-x_3 != 0:
        m2 = (y_4-y_3)/(x_4-x_3)
    else:
        m2 = 2147483647
    m_1 = (y_2-y_1)/(x_2-x_1)
    m_2 = (y_4-y_3)/(x_4-x_3)
    b_1 = y_1 - m_1*x_1
    b_2 = y_3 - m_2*x_3
    x = (b_2-b_1)/(m_1-m_2)
    y = m_1*x + b_1


    return (int(x),int(y))
    
    #cv2.circle(poi,(int(x/2),int(avY)),5,(0,0,255),-1)
    #print(avY)
    #cv2.imshow('poi',poi)

def calculateCorners(img,x1,y1,x2,y2,x3,y3,x4,y4):

    x_1,y_1 = findEdgePoint(img,x1,y1)
    x_2,y_2 = findEdgePoint(img,x2,y2)
    x_3,y_3 = findEdgePoint(img,x3,y3)
    x_4,y_4 = findEdgePoint(img,x4,y4)

    if x_2-x_1 != 0:
        m_1 = (y_2-y_1)/(x_2-x_1)
    else:
        m_1 = 21474836
    if x_4-x_3 != 0:
        m_2 = (y_4-y_3)/(x_4-x_3)
    else:
        m_2 = 21474836
    #m_1 = (y_2-y_1)/(x_2-x_1)
    #m_2 = (y_4-y_3)/(x_4-x_3)
    b_1 = y_1 - m_1*x_1
    b_2 = y_3 - m_2*x_3
    x = (b_2-b_1)/(m_1-m_2)
    y = m_1*x + b_1

    xLower = x + 1695.2*np.cos(np.arctan(m_1))
    yLower = y + 1695.2*np.sin(np.arctan(m_1))
    
    return (int(x),int(y)),(int(xLower),int(yLower))
    
# #Start Picam
picam2 = Picamera2()

preview_config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 320)})
capture_config = picam2.create_still_configuration(main={"format": 'XRGB8888', "size": (4656, 3496)})
picam2.configure(capture_config)
picam2.start()

focuser = Focuser('/dev/v4l-subdev1')
focuser.set(Focuser.OPT_FOCUS,750)

while True:

    og = picam2.capture_array()
    og = cv2.cvtColor(og, cv2.COLOR_BGRA2BGR)
    
  
    #findEdgePoint(og,2040,1130,40,110)

    
   

    
    

    
    x1,y1 = 2431,2715 #bottom left
    x2,y2 = 2237,1173 #top left left
    x3,y3 = 2300,1095  #top left right
    x4,y4 = 3000,1043 #top right left
    x5,y5 = 3075,1080 #bottom right
    x6,y6 = 3173,2700 #top right right

    BLCorner,FLCorner = calculateCorners(og,x1,y1,x2,y2,x3,y3,x4,y4)
    BRCorner,FRCorner = calculateCorners(og,x5,y5,x6,y6,x3,y3,x4,y4)
    
    #findEdgePoint(og,x1,y1)
    #findEdgePoint(og,x2,y2)
    # findEdgePoint(og,x3,y3)
    # findEdgePoint(og,x4,y4)
    # findEdgePoint(og,x5,y5)
    # findEdgePoint(og,x6,y6)

    cv2.line(og,BLCorner,BRCorner,(0,0,255),8)
    cv2.line(og,BLCorner,FLCorner,(0,0,255),8)
    cv2.line(og,BRCorner,FRCorner,(0,0,255),8)
    cv2.line(og,FRCorner,FLCorner,(0,0,255),8)

    cv2.circle(og,findEdgePoint(og,x1,y1),7,(0,255,0),-1)
    cv2.circle(og,findEdgePoint(og,x2,y2),7,(0,255,0),-1)
    cv2.circle(og,findEdgePoint(og,x3,y3),7,(0,255,0),-1)
    cv2.circle(og,findEdgePoint(og,x4,y4),7,(0,255,0),-1)
    cv2.circle(og,findEdgePoint(og,x5,y5),7,(0,255,0),-1)
    cv2.circle(og,findEdgePoint(og,x6,y6),7,(0,255,0),-1)

    og = cv2.resize(og,(1500,1126))
    cv2.imshow('og',og)
    
    if cv2.waitKey(1) == ord('q'):
        break
picam2.close()
cv2.destroyAllWindows()
