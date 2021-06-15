# Import essential libraries
import requests
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream

url = "http://101.95.190.13:8989/shot.jpg"
vs = VideoStream(0).start()

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=640, height=480)
    image = vs.read()
    cv2.line(image, pt1=(320, 0), pt2=(320, 480), color=(255, 0, 0),
             thickness=2, lineType=8, shift=0)
    cv2.line(image, pt1=(0, 240), pt2=(640, 240), color=(255, 0, 0),
             thickness=2, lineType=8, shift=0)
    cv2.line(img, pt1=(320, 0), pt2=(320, 480), color=(255, 0, 0),
             thickness=2, lineType=8, shift=0)
    cv2.line(img, pt1=(0, 240), pt2=(640, 240), color=(255, 0, 0),
             thickness=2, lineType=8, shift=0)
    cv2.imshow('Web_Cam', image)
    cv2.imshow("Android_cam", img)


    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
