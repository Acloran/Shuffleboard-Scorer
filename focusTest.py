from arducam.focus.Focuser import Focuser
from picamera2 import Picamera2
import numpy as np 
import cv2

picam2 = Picamera2()

preview_config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 320)})
capture_config = picam2.create_still_configuration(main={"format": 'XRGB8888', "size": (4656, 3496)})
picam2.configure(preview_config)
picam2.start()


focuser = Focuser('/dev/v4l-subdev1')
focuser.set(Focuser.OPT_FOCUS,750)

while(1):
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break