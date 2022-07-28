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

    bgr_low = np.array([159, 0, 0], np.uint8)
    bgr_high = np.array([226, 208, 153], np.uint8)
	
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

def findEdgePointY(img,x1,y1):
    width = 40
    height = 100
    halfWidth = int(width/2)
    halfHeight = int(height/2)

    tlX = x1-halfWidth
    tlY = y1-halfHeight

    poi = img[tlY:tlY+height,tlX:tlX+width]
   
    processedImg = recolorImg(poi)
    
    #(y,x) = processedImg.shape
    leftY = halfHeight
    rightY = halfHeight
    for i in range(0,height-1):
        if processedImg.item(i,0)==255:
            leftY = i
            break

    for i in range(0,height-1):
        if processedImg.item(i,width-1)==255:
            rightY = i
            break

    avY = (leftY+rightY)/2.0
    
    #cv2.circle(poi,(int(x/2),int(avY)),5,(0,0,255),-1)
    #print(avY)
    #cv2.imshow('poi',poi)
    outX = tlX+halfWidth
    outY = tlY+int(avY)
    return (outX, outY)

def findEdgePointY(img,x1,y1):
    width = 100
    height = 40
    halfWidth = int(width/2)
    halfHeight = int(height/2)

    tlX = x1-halfWidth
    tlY = y1-halfHeight

    poi = img[tlY:tlY+height,tlX:tlX+width]
   
    processedImg = recolorImg(poi)
    
    #(y,x) = processedImg.shape
    topX = halfWidth
    bottomX = halfWidth
    for i in range(0,width-1):
        if processedImg.item(0,i)==255:
            topX = i
            break

    for i in range(0,width-1):
        if processedImg.item(height-1,i)==255:
            bottomX = i
            break

    avX = (topX+bottomX)/2.0
    
    #cv2.circle(poi,(int(x/2),int(avY)),5,(0,0,255),-1)
    #print(avY)
    cv2.imshow('poi',poi)
    outX = tlX+int(avX)
    outY = tlY+halfHeight
    return (outX, outY)

#Start Picam
picam2 = Picamera2()

preview_config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 320)})
capture_config = picam2.create_still_configuration(main={"format": 'XRGB8888', "size": (4656, 3496)})
picam2.configure(capture_config)
picam2.start()


while True:

    og = picam2.capture_array()
    og = cv2.cvtColor(og, cv2.COLOR_BGRA2BGR)
    
  
    #findEdgePoint(og,2040,1130,40,110)

    
   

    
    


    print('backpoint:', findEdgePointY(og,2064,1188))
    print('leftpoint:', findEdgePointY(og,1980,1224))

    if cv2.waitKey(1) == ord('q'):
        break
picam2.close()
cv2.destroyAllWindows()
