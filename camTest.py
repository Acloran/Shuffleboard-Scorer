#from doctest import OutputChecker
import numpy as np 
import cv2
import time
import picamera
#import keyboard

#cam = cv2.VideoCapture(0)#,cv2.CAP_DSHOW)

#cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # set new dimensionns to cam object (not cap)
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)

while True:
    with picamera.PiCamera() as camera:
        camera.resolution = (1920, 1080)
        camera.framerate = 3
        time.sleep(2)
        output = np.empty((1080, 1920, 3), dtype=np.uint8)
        camera.capture(output, 'bgr')
        image = image.reshape((1080, 1920, 3))
        #ret, img = cam.read()
    cv2.imshow('result',image)

    #cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        while True:
            if cv2.waitKey(1) == ord('x'):
                break
        break

cv2.destroyAllWindows()