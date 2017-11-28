#!/usr/bin/python2
# DO: chmod +x webcam.py

import time
from SimpleCV import Camera # Install: sudo pip install simplecv, python2


cam = Camera()
time.sleep(0.1)
img = cam.getImage()
img.save("simplecv.png")
