import numpy as np
import cv2

#cv2.namedWindow("output", cv2.WINDOW_NORMAL)

face_cascade = cv2.CascadeClassifier('data/face.xml')
#eye_cascade = cv2.CascadeClassifier('data/eye.xml')
img = cv2.imread('pictures/people.jpg')

rgb_gbr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(rgb_gbr, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #imS = cv2.resize(img, (x, y))
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    cv2.imshow('img', roi_color)
    cv2.waitKey(0)
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    #for (ex,ey,ew,eh) in eyes:
    #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

#imS = cv2.resize(img, (400, 445))

cv2.imshow('img', img)
cv2.waitKey(0)
#cv2.destroyAllWindows()
