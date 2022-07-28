from doctest import OutputChecker
import numpy as np 
import cv2
import keyboard
from picamera2 import Picamera2

# cam = cv2.VideoCapture(0)#,cv2.CAP_DSHOW)

# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensionns to cam object (not cap)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

#cam = cv2.imread('images/c1.png')
outWidth = 800
outHeight = 1600

class Puck:
    def __init__(self, xVal, yVal, isBlue, idNum, scoreVal):
        self.isBlue = isBlue
        self.yVal = yVal
        self.xVal = xVal
        self.idNum = idNum
        self.score = scoreVal
    def getXVal(self):
        return self.xVal
    def getYVal(self):
        return self.yVal
    def getIsBlue(self):
        return self.isBlue
    def getIDNum(self):
        return self.idNum
    def getScore(self):
        return self.score

def drawCircles(puckList, tableImg):
    for obj in puckList:
        if obj.getIsBlue():
            circleColor = (255,0,0)
        else:
            circleColor = (0,0,255)
        
        cv2.circle(tableImg,(int(obj.getXVal()),int(obj.getYVal())),47,circleColor,6)
        #cv2.putText(imgout, str(obj.getIDNum()), (int(obj.getXVal())+52,int(obj.getYVal())+14), cv2.FONT_HERSHEY_SIMPLEX, 1.5, circleColor, 2, cv2.LINE_AA)
        cv2.circle(tableImg,(int(obj.getXVal()),int(obj.getYVal())),20,circleColor,-1)

    # puckRadius = 50
    # xVal = int(x)
    # yVal = int(y)
    # colormask = np.zeros((outHeight,outWidth,1), np.uint8)
    # cv2.circle(colormask,(int(x),int(y)),35,(255),30)
    
    # ret, mask = cv2.threshold(colormask, 10, 255, cv2.THRESH_BINARY)
    # result = cv2.bitwise_and(imgin, imgin, mask=mask)
    # r = result[:,:,2]

    # data = r[np.nonzero(r)]
    # means = np.mean(data, axis=None)

    # redVal = round(means, 1)
    # strRedVal = str(redVal)

    # if redVal < 130:
    #     circleColor = (255,0,0)
    #     isBlue = True
    # else:
    #     circleColor = (0,0,255)
    #     isBlue = False

    # cv2.putText(imgout, str(int(y)), (xVal+52,yVal+14), cv2.FONT_HERSHEY_SIMPLEX, 1.5, circleColor, 2, cv2.LINE_AA)
    # cv2.circle(imgout,(int(x),int(y)),47,circleColor,6)
    # # draw the center of the circle
    # cv2.circle(imgout,(int(x),int(y)),20,circleColor,-1)
    return tableImg

def isRedorBlue(imgin, x, y):
    puckRadius = 50
    xVal = int(x)
    yVal = int(y)
    colormask = np.zeros((1600,800,1), np.uint8)
    cv2.circle(colormask,(int(x),int(y)),35,(255),30)
    
    ret, mask = cv2.threshold(colormask, 10, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(imgin, imgin, mask=mask)
    # cv2.imshow('mask??', mask)
    # cv2.imshow('colormask', colormask)
    # cv2.imshow('raw', imgin)
    r = result[:,:,2]

    data = r[np.nonzero(r)]
    means = np.mean(data, axis=None)

    redVal = round(means, 1)
    strRedVal = str(redVal)

    if redVal < 130:
        circleColor = (255,0,0)
        isBlue = True
    else:
        circleColor = (0,0,255)
        isBlue = False

    #cv2.putText(imgout, str(int(y)), (xVal+52,yVal+14), cv2.FONT_HERSHEY_SIMPLEX, 1.5, circleColor, 2, cv2.LINE_AA)
    #cv2.circle(imgout,(int(x),int(y)),47,circleColor,6)
    # draw the center of the circle
    #cv2.circle(imgout,(int(x),int(y)),20,circleColor,-1)
    return isBlue



def drawTable(line1, line2, line3):
    #define an all black image
    outputImg = np.zeros((1600,800,3), np.uint8)
    #draw scoring lines and numbers
    lineThick = 8
    cv2.line(outputImg,(0,line1),(800,line1),(0,255,0),lineThick)
    cv2.line(outputImg,(0,line2),(800,line2),(0,255,0),lineThick)
    cv2.line(outputImg,(0,line3),(800,line3),(0,255,0),lineThick)
    
    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 5
    fontThickness = 5
    cv2.putText(outputImg,'1',(365,line1+175), font, fontScale,(0,255,0),fontThickness,cv2.LINE_AA)
    cv2.putText(outputImg,'2',(365,line2+175), font, fontScale,(0,255,0),fontThickness,cv2.LINE_AA)
    cv2.putText(outputImg,'3',(365,line3+175), font, fontScale,(0,255,0),fontThickness,cv2.LINE_AA)
    return outputImg

def resizeImg(img, width, height):
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

def warpImg(img):
    pts1 = np.float32([[2010,1186],[2813,1209],[2774,2912],[1940,2878]])
    pts2 = np.float32([[0,0],[outWidth,0],[outWidth,outHeight],[0,outHeight]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(outWidth,outHeight))
    return dst

def recolorImg(img):
    frame = cv2.bitwise_not(img)

    bgr_low = np.array([135, 99, 61], np.uint8)
    bgr_high = np.array([227, 172, 132], np.uint8)
	
    greenMask = cv2.inRange(frame, bgr_low, bgr_high)
    greenMask = cv2.medianBlur(greenMask,7)

    kernel = np.ones((9,9),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)

    closing = cv2.morphologyEx(greenMask, cv2.MORPH_CLOSE, kernel)
    ret,thresh = cv2.threshold(closing,127,255,0)
    return thresh

def processImg(img):
    res = warpImg(img)
    res = recolorImg(res)
    return res

def findContours(imgCountours, imgRaw):
    
    contours,hierarchy = cv2.findContours(imgCountours, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
    #smallthresh = resizeImg(thresh, 300, 600)
    #cv2.imshow('mask',smallthresh) 
    pucks = []

    for i in contours[:]:
        (x,y),radius = cv2.minEnclosingCircle(i)
        center = (int(x),int(y))
        radius = int(radius)
       #scoring breakdown
        if radius>15 and radius<30 and cv2.contourArea(i)>300:
            if y<42:
                score = 4
            elif y<208:
                score = 3
            elif y<487:
                score = 2
            elif y<1552:
                score = 1
            else: 
                score = 0

            pucks.append(Puck(int(x),int(y),isRedorBlue(imgRaw,x, y),i,score))
    return pucks

def sortPucks(pucklist):
    pucklist.sort(key=lambda x: x.getYVal())
    return pucklist

def scorePucks(sortedPuckList):
    scoringPucks = []
    if len(sortedPuckList) > 0:        
        scoreIsBlue = sortedPuckList[0].getIsBlue()
        runningScore = 0
        for obj in sortedPuckList:
            if obj.getIsBlue()==scoreIsBlue:
                runningScore += obj.getScore()
            else: break
        if scoreIsBlue:
            return (0,runningScore)
        else:
            return (runningScore,0)

#Start Picam
picam2 = Picamera2()

preview_config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 320)})
capture_config = picam2.create_still_configuration(main={"format": 'XRGB8888', "size": (4656, 3496)})
picam2.configure(capture_config)
picam2.start()


while True:

    #define an all black image
    outputImg = drawTable(546,271,0)
    
    #get the image from the camera
    
    og = picam2.capture_array()
    og = cv2.cvtColor(og, cv2.COLOR_BGRA2BGR)
    #straighten image
    #og = cam
    og = og[ 800:3000,1700:3100]
    
    #pts1 = np.float32([[2010,1186],[2813,1209],[2774,2912],[1940,2878]])
    #pts2 = np.float32([[0,0],[outWidth,0],[outWidth,outHeight],[0,outHeight]])

    #M = cv2.getPerspectiveTransform(pts1,pts2)

    #dst = cv2.warpPerspective(og,M,(outWidth,outHeight))
    
    #resized = resizeImg(dst, 300, 600)
    #cv2.imshow('resized', resized)
    #recolor Image
    frame = cv2.bitwise_not(og)
    #frame = og
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    


    bgr_low = np.array([0, 49,120], np.uint8)
    bgr_high = np.array([138, 120, 255], np.uint8)

	
    greenMask = cv2.inRange(frame, bgr_low, bgr_high)
    greenMask = cv2.medianBlur(greenMask,7)

    kernel = np.ones((9,9),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)

    closing = cv2.morphologyEx(greenMask, cv2.MORPH_CLOSE, kernel)
 
    
    ret,thresh = cv2.threshold(closing,127,255,0)

    #find corner testing:
    
    #gray = np.float32(gray)
    #dst = cv2.cornerHarris(thresh,2,3,0.04)
    #result is dilated for marking the corners, not important
    #dst = cv2.dilate(dst,kernel2)
    # Threshold for an optimal value, it may vary depending on the image.
    #frame[dst>0.01*dst.max()]=[0,0,255]

    circles = cv2.HoughCircles(thresh,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(og,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(og,(i[0],i[1]),2,(0,0,255),3)
    
    # contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)

    # for i in contours[:]:
    #     (x,y),radius = cv2.minEnclosingCircle(i)
    #     center = (int(x),int(y))
    #     radius = int(radius)
    #     if radius<20 and radius>10:
    #         cv2.circle(greenMask,center,radius,(0,255,0),3)
            
    #smallthresh = resizeImg(thresh, 300, 600)
    #cv2.imshow('mask',smallthresh) 
    pucks = []
    smallframe = resizeImg(og, 763,1200)
    cv2.imshow('dst',smallframe)
    
    smallImg = resizeImg(outputImg, 450, 900)
    #cv2.imshow('result',smallImg)        
    #cv2.imshow('result',outputImg)
    #resized = cv2.resize(outputImg, dim, interpolation = cv2.INTER_AREA)
    #cv2.imshow('result',resized)
    # sortedPucks = []


    if cv2.waitKey(500) == ord('q'):
        break
picam2.close()
cv2.destroyAllWindows()
