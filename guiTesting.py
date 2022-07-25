from doctest import OutputChecker
import numpy as np 
import cv2
import keyboard
from main import *

cv2.namedWindow('ShuffleScore',cv2.WINDOW_FULLSCREEN)

redColor = (0,0,255)
blueColor = (255,0,0)
greenColor = (0,255,0)

blueScore = 0
redScore = 0

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

buttonLocations = [(150,200,200,100,'Increase',greenColor,leftUpButton),(150,600,200,100,'Decrease',greenColor,leftDownButton),(1150,200,200,100,'Increase',greenColor,rightUpButton),(1150,600,200,100,'Decrease',greenColor,rightDownButton)]



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
        if onButton:
            print(redScore)

def draw_buttons():
    for (x1,y1,w,h,string,color,funct) in buttonLocations:
        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),color,-1)
        cv2.putText(img,string,((x1+int(w/2)-75),(y1+int(h/2)+10)),cv2.FONT_HERSHEY_DUPLEX,1.1,(0,0,0),1,cv2.LINE_AA)


cv2.setMouseCallback('ShuffleScore',register_buttons)


#draw a sample output from the cam

sampleOut = resizeImg(drawTable(546,271,0),490,900)
sampleOut = cv2.copyMakeBorder(sampleOut,0,0,5,5,cv2.BORDER_CONSTANT,value=greenColor)
img = np.zeros((900,1500,3), np.uint8)



while(1):
    img = np.zeros((900,1500,3), np.uint8)

    draw_buttons()
    if redScore > 9:
        redNumX = 90
    else:
        redNumX = 165

    if blueScore > 9:
        blueNumX = 1090
    else:
        blueNumX = 1165
    
    cv2.putText(img,str(redScore),((redNumX),(528)),cv2.FONT_HERSHEY_DUPLEX,8,redColor,10,cv2.LINE_AA)
    cv2.putText(img,str(blueScore),((blueNumX),(528)),cv2.FONT_HERSHEY_DUPLEX,8,blueColor,10,cv2.LINE_AA)


    img[0:900, 500:1000] = sampleOut
    

    cv2.imshow('ShuffleScore',img)
    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('q'):
        break
    elif pressedKey == ord('r'):
        img = np.zeros((900,1500,3), np.uint8)

cv2.destroyAllWindows()