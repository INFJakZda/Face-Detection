# PYTHON 2 MUST BEE !!!!

import os
import numpy as np
import cv2

import time
from SimpleCV import Camera # Install: sudo pip install simplecv, python2


face_cascade = cv2.CascadeClassifier('data/face.xml')

cam = Camera()
while True:
    time.sleep(0.1)
    img = cam.getImage()
    img.save("simplecv.png")

    img = cv2.imread('simplecv.png')
    rgb_gbr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(rgb_gbr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    #cv2.imwrite('simplecv.png', img)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey(200)
    cv2.destroyAllWindows()
    
    os.remove("simplecv.png")
    time.sleep(0.5)
