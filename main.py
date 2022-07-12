from doctest import OutputChecker
import numpy as np 
import cv2
import keyboard

cam = cv2.VideoCapture(0)#,cv2.CAP_DSHOW)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensionns to cam object (not cap)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

#cam = cv2.imread('images/c1.png')


def drawRedorBlueCircle(imgin, imgout, x, y):
    xVal = int(x)
    yVal = int(y)
    colormask = np.zeros((300,620,1), np.uint8)
    cv2.circle(colormask,(int(x),int(y)),13,(255),10)
    
    ret, mask = cv2.threshold(colormask, 10, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(imgin, imgin, mask=mask)
    r = result[:,:,2]

    data = r[np.nonzero(r)]
    means = np.mean(data, axis=None)

    redVal = round(means, 1)
    strRedVal = str(redVal)

    if redVal < 180:
        circleColor = (255,0,0)
        isBlue = True
    else:
        circleColor = (0,0,255)
        isBlue = False

    #cv2.putText(imgout, strRedVal, (xVal+22,yVal+5), cv2.FONT_HERSHEY_SIMPLEX, .5, circleColor, 2, cv2.LINE_AA)
    cv2.circle(imgout,(int(x),int(y)),18,circleColor,2)
    # draw the center of the circle
    cv2.circle(imgout,(int(x),int(y)),6,circleColor,-1)
    return isBlue
    
class Puck:
    def __init__(self, xVal, isBlue, idNum, scoreVal):
        self.isBlue = isBlue
        self.xVal = xVal
        self.idNum = idNum
        self.scoreVal = scoreVal
    
    def getXVal(self):
        return self.xVal
    def getIsBlue(self):
        return self.isBlue
    def getIDNum(self):
        return self.idNum

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

def drawTable(line1, line2, line3):
    #define an all black image
    outputImg = np.zeros((300,620,3), np.uint8)
    #draw scoring lines and numbers
    cv2.line(outputImg,(line1,0),(line1,300),(0,255,0),2)
    cv2.line(outputImg,(line2,0),(line2,300),(0,255,0),2)
    cv2.line(outputImg,(line3,0),(line3,300),(0,255,0),2)
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(outputImg,'1',(line1-57,165), font, 2.2,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(outputImg,'2',(line2-72,165), font, 2.2,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(outputImg,'3',(line3-75,165), font, 2.2,(0,255,0),2,cv2.LINE_AA)
    return outputImg

while True:

    #define an all black image
    outputImg = drawTable(392,498,600)


    ret, og = cam.read()
    
    #straighten image
    #og = cam
    
    pts1 = np.float32([[297,317],[734,319],[732,541],[292,528]])
    pts2 = np.float32([[0,0],[600,0],[600,300],[0,300]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(og,M,(620,300))
    #cv2.imshow('raw',dst) 
    #recolor Image
    frame = cv2.bitwise_not(dst)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    


    bgr_low = np.array([30, 0, 59], np.uint8)
    bgr_high = np.array([210, 135, 255], np.uint8)

	
    greenMask = cv2.inRange(frame, bgr_low, bgr_high)
    greenMask = cv2.medianBlur(greenMask,7)

    kernel = np.ones((7,7),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)

    closing = cv2.morphologyEx(greenMask, cv2.MORPH_CLOSE, kernel)
 
    
    ret,thresh = cv2.threshold(closing,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE) 
  
    puckPos = []

    for i in contours[:]:
        (x,y),radius = cv2.minEnclosingCircle(i)
        center = (int(x),int(y))
        radius = int(radius)
       
        if radius>4 and radius<15:
            if x>600:
                score = 4
            elif x>518:
                score = 3
            elif x>412:
                score = 2
            elif x>20:
                score = 1
            else: 
                score = 0

            puckPos.append(Puck(int(x),drawRedorBlueCircle(dst, outputImg, x, y),i,score))

            
    cv2.imshow('result',outputImg)

    sortedPuckPos = []

    while len(puckPos)>0:
        maximum = puckPos[0]
        for obj in puckPos:
            if obj.getXVal()>maximum.getXVal():
                maximum = obj
                sortedPuckPos.append(obj)
                puckPos.remove(obj)
                print('passed')
                

    for obj in sortedPuckPos:
        if obj.getIsBlue():
            print('Blue puck at ' + str(obj.getXVal()))
        else:
            print('Red puck at ' + str(obj.getXVal()))
        

    if cv2.waitKey(400) == ord('q'):
        while True:
            if cv2.waitKey(1) == ord('x'):
                break
        break
cam.release()
cv2.destroyAllWindows()