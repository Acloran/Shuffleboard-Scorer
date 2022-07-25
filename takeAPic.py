from doctest import OutputChecker
import numpy as np 
import cv2
import keyboard
from picamera2 import Picamera2


# cam = cv2.VideoCapture(0)#,cv2.CAP_DSHOW)

# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensionns to cam object (not cap)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

# ret,frame = cam.read()
#Start Picam
picam2 = Picamera2()

preview_config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 320)})
capture_config = picam2.create_still_configuration(main={"format": 'XRGB8888', "size": (4656, 3496)})
picam2.configure(capture_config)
picam2.start()

frame = picam2.capture_array()

#focuser.set(Focuser.OPT_FOCUS,800)
outWidth = 800
outHeight = 1600

while(True):
     #display the captured image
    #print(frame.shape)
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    #frame = cv2.imread('../Local Testing/tester.png')
    #straighten image
    #og = cam
    #cv2.imshow('img1',frame)

    pts1 = np.float32([[2010,1186],[2813,1209],[2774,2912],[1940,2878]])
    pts2 = np.float32([[0,0],[outWidth,0],[outWidth,outHeight],[0,outHeight]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(frame,M,(outWidth,outHeight))
    cv2.imshow('img2',dst)
    if cv2.waitKey(500) & 0xFF == ord('q'): #save on pressing 'y' 
        
        cv2.imwrite('images/c1.png',frame)
        cv2.imwrite('images/c2.png',dst)
        cv2.destroyAllWindows()
        break

picam2.close()
