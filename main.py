# Import essential libraries
import requests
import cv2
import numpy as np
import imutils
import pyttsx3
from imutils.video import VideoStream
import warnings

warnings.filterwarnings("ignore")
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"
# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


def direction(image):
    while True:
        H, W = image.shape[:2]
        # image = cv2.flip(image, 1)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                                0.3)

        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # To find Centre of Object
                m = int(x + w / 2)
                n = int(y + h / 2)
                Line_Position1 = int(image.shape[1] * (50 / 100))

                cv2.line(image, pt1=(Line_Position1, 0), pt2=(Line_Position1, image.shape[0]), color=(255, 0, 0),
                         thickness=2, lineType=8, shift=0)

                bounding_mid = (int(m), int(n))
                if (bounding_mid):
                    cv2.line(img=image, pt1=bounding_mid, pt2=(Line_Position1, bounding_mid[1]), color=(255, 0, 0),
                             thickness=1, lineType=8, shift=0)
                distance_from_line_x = bounding_mid[0] - Line_Position1
                Line_Position2 = int(image.shape[0] * (50 / 100))

                cv2.line(image, pt1=(0, Line_Position2), pt2=(image.shape[1], Line_Position2), color=(255, 0, 0),
                         thickness=2, lineType=8, shift=0)

                bounding_mid = (int(m), int(n))
                if (bounding_mid):
                    cv2.line(img=image, pt1=bounding_mid, pt2=(bounding_mid[0], Line_Position2), color=(255, 0, 0),
                             thickness=1, lineType=8, shift=0)
                distance_from_line_y = Line_Position2 - bounding_mid[1]
                return int(distance_from_line_x), int(distance_from_line_y)

        cv2.imshow('Frame', image)
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break


# Web-Cam Reading
vs = VideoStream(0).start()
image = vs.read()
xrm, yrm = direction(image)

# Mobile Camera Reading
url = "http://101.95.190.13:8989/shot.jpg"  # Put your camera IP address followed by /shot.jpg

img_resp = requests.get(url)
img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
img = cv2.imdecode(img_arr, -1)
img = imutils.resize(img, width=640, height=480)
xlm, ylm, = direction(img)

# t is the distance between the centre of two camera

t = 13.7

height, width = image.shape[:2]

alpha = 56.4  # find alpha using CamCalibration.py


# To find depth of image
def Depth(xlm, xrm):
    f_pixel = (width * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)

    d = ((f_pixel * t) / (xlm - xrm))
    return abs(d)


depth = Depth(xlm, xrm)
depth = round(depth, 4)

engine = pyttsx3.init()


def talk(text):
    engine.say(text)
    engine.runAndWait()


talk(f"the object distance is {depth} centimeter")
