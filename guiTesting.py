from doctest import OutputChecker
import numpy as np 
import cv2
import keyboard
from main import *
from picamera2 import Picamera2

cv2.namedWindow('ShuffleScore',cv2.WINDOW_FULLSCREEN)

redColor = (0,0,255)
blueColor = (255,0,0)
greenColor = (0,255,0)

blueScore = 0
redScore = 0

tempBlueScore = 0
tempRedScore = 0

picam2 = Picamera2()

def leftUpButton():
    global redScore
    redScore += 1

def leftDownButton():
    global redScore
    redScore -= 1

def rightUpButton():
    global blueScore
    blueScore += 1

def rightDownButton():
    global blueScore
    blueScore -= 1

def scoreTable():
    global redScore
    global blueScore
    #(r,b) = score
    redScore += liveRed
    blueScore += liveBlue

buttonLocations = [(150,100,200,100,'Increase',greenColor,leftUpButton),(150,500,200,100,'Decrease',greenColor,leftDownButton),(1150,100,200,100,'Increase',greenColor,rightUpButton),(1150,500,200,100,'Decrease',greenColor,rightDownButton),(625,800,250,100,'Run Score',greenColor,scoreTable)]
def startPiCam2():
    preview_config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 320)})
    capture_config = picam2.create_still_configuration(main={"format": 'XRGB8888', "size": (4656, 3496)})
    picam2.configure(capture_config)
    picam2.start()

def getImage():
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def register_buttons(event,x,y,flags,param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        count = 0
        onButton = False
        for (x1,y1,w,h,string,color,funct) in buttonLocations:
            if x1 <= x <= x1+w and y1 <= y <= y1+h:
                onButton = True
                funct()
                break
            count+=1
        # if onButton:
        #     print(redScore)

def draw_buttons():
    for (x1,y1,w,h,string,color,funct) in buttonLocations:
        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),color,-1)
        cv2.putText(img,string,((x1+28),(y1+int(h/2)+10)),cv2.FONT_HERSHEY_DUPLEX,1.1,(0,0,0),1,cv2.LINE_AA)

def draw_scores():
    if redScore > 9:
        redNumX = 90
    else:
        redNumX = 165

    if blueScore > 9:
        blueNumX = 1090
    else:
        blueNumX = 1165
    
    cv2.putText(img,str(redScore),((redNumX),(428)),cv2.FONT_HERSHEY_DUPLEX,8,redColor,10,cv2.LINE_AA)
    cv2.putText(img,str(blueScore),((blueNumX),(428)),cv2.FONT_HERSHEY_DUPLEX,8,blueColor,10,cv2.LINE_AA)
    cv2.putText(img,str(tempRedScore),((redNumX),(858)),cv2.FONT_HERSHEY_DUPLEX,8,greenColor,8,cv2.LINE_AA)
    cv2.putText(img,str(tempBlueScore),((blueNumX),(858)),cv2.FONT_HERSHEY_DUPLEX,8,greenColor,8,cv2.LINE_AA)

cv2.setMouseCallback('ShuffleScore',register_buttons)


#draw a sample output from the cam




startPiCam2()

while(1):
    realOut = drawTable(546,271,0)
    
    capture = getImage()
    raw = warpImg(capture)
    dst = processImg(capture)

    myPuckList = findContours(dst,raw)
    
    score = scorePucks(sortPucks(myPuckList))
    
    realOut = drawCircles(myPuckList, realOut)
    

    img = np.zeros((900,1500,3), np.uint8)
    #cv2.imshow('out',realOut)
    realOut = resizeImg(realOut,490,900)
    realOut = cv2.copyMakeBorder(realOut,0,0,5,5,cv2.BORDER_CONSTANT,value=greenColor)
    img[0:900, 500:1000] = realOut

    if score != None:
        (liveRed, liveBlue) = score
    else:
        (liveRed, liveBlue) = (0,0)
    tempBlueScore = blueScore + liveBlue
    tempRedScore = redScore + liveRed

    draw_buttons()

    draw_scores()

    
    

    cv2.imshow('ShuffleScore',img)
    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('q'):
        break
    elif pressedKey == ord('r'):
        img = np.zeros((900,1500,3), np.uint8)
picam2.close()
cv2.destroyAllWindows()