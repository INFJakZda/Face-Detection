import cv2
import os
import subprocess
import time
import numpy as np

import pygame
import pygame.camera
import pygame.image
from pygame.locals import *

from PIL import Image

import pickle

import math

subjects = ["", "John_Travolta", "Julianne_Moore", "Salma_Hayek", "Silvio_Berlusconi", "a", "b", "c", "d"]


def detect_face(img):   #z obrazka wycina twarze kolorowe i ich współrzedne i zwraca w liscie [(twarz, wspolrzedne), ....]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('data/face.xml')
    cord_list = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    faces = []  

    if (len(cord_list) == 0):
        return None, None

    for it in range(len(cord_list)):
        (x, y, w, h) = cord_list[it]
        faces.append(gray[y:y+w, x:x+h])

    return faces, cord_list


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    
    if os.path.isfile('training-data/ptd_faces.data'):
        with open('training-data/ptd_faces.data', 'rb') as f:
            faces = pickle.load(f)
        with open('training-data/ptd_labels.data', 'rb') as f:
            labels = pickle.load(f)    
    else:
        for dir_name in dirs:
            if not dir_name.startswith("s"):
                continue;
            label = int(dir_name.replace("s", ""))
            subject_dir_path = data_folder_path + "/" + dir_name
            subject_images_names = os.listdir(subject_dir_path)
            for image_name in subject_images_names:
                if image_name.startswith("."):
                    continue;
                image_path = subject_dir_path + "/" + image_name
                image = cv2.imread(image_path)
                face, rect = detect_face(image)

                if face is not None:
                    faces.append(face[0])
                    labels.append(label)
                
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        
        with open('training-data/ptd_faces.data', 'wb+') as f:
            pickle.dump(faces, f)
        with open('training-data/ptd_labels.data', 'wb+') as f:
            pickle.dump(labels, f)
        
    
    return faces, labels

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 0.3, (0, 255, 0), 1)

def predict(test_img):
    img = test_img.copy()
    faces, rect = detect_face(img)
    if(faces != None):
        for it in range(len(faces)):
            label, confidence = face_recognizer.predict(faces[it])
            label_text = subjects[label]
        
            draw_rectangle(img, rect[it])
            draw_text(img, label_text, rect[it][0], rect[it][1]-5)

    return img

def detect_eye(image, cords):   #na całym obrazku rysuje obrys oczu dla kazdej wykrytej twarzy

    (x, y, w, h) = cords
    
    face = image.copy()      
    face[y:y+int(w/3.3), x:x+h] = (255,255,255)
    face[y+int(w/2.1):y+w, x:x+h] = (255,255,255)
    face[y:y+w, x:x+int(h/5)] = (255,255,255)
    face[y:y+w, x+(h-int(h/5)):x+h] = (255,255,255)  

    imgray = cv2.cvtColor(face[y:y+w, x:x+h], cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    image2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    image[y:y+w, x:x+h] = cv2.drawContours(image[y:y+w, x:x+h], contours, -1, (100,255,100), 1)

    #cv2.imshow("image",  image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return image     

def hand_seek(image):
    cv2.rectangle(image, (300,300), (100,100), (255,243,58),0)
    crop_img = image[100:300, 100:300]
    
    img_grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_grey, (35,35), 0)
    
    # thresholdin: Otsu's Binarization method
    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    image2, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # find contour with max area
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # create bounding rectangle around the contour (can skip below two lines)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt)

    # drawing contours
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)

    # finding convexity defects
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
    
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignore angles > 90 and highlight rest with red dots
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0,0,255], -1)
        #dist = cv2.pointPolygonTest(cnt,far,True)

        # draw a line from start to end i.e. the convex points (finger tips)
        # (can skip this part)
        cv2.line(crop_img,start, end, [0,255,0], 2)
    
    return image

if __name__ == '__main__':
    print("Preparing data...")
    faces, labels = prepare_training_data("training-data")
    print("Data prepared")

    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))

    pygame.init()
    pygame.camera.init()
    display = pygame.display.set_mode((640, 480), 0)
    cam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
    cam.start()
    screen = pygame.surface.Surface((640, 480), 0, display)
    capture = True

    while capture:
        screen = cam.get_image(screen)

        pil_img = pygame.image.tostring(screen, "RGBA", False)
        test_img = Image.frombytes("RGBA",(640,480), pil_img)
        test_img = cv2.cvtColor(np.array(test_img), cv2.COLOR_RGB2BGR)
       
        #predicted_img = predict(test_img)
        faces, cords = detect_face(test_img)
        if(faces != None):
            for it in range(len(faces)):
                test_img = detect_eye(test_img, cords[it])
        elif(faces == None):
            hand_seek(test_img)
        
        test_img = cv2.cvtColor(np.array(test_img), cv2.COLOR_BGR2RGB)
        
        #img = pygame.image.frombuffer(predicted_img.tostring(), predicted_img.shape[1::-1], "RGB")
        img = pygame.image.frombuffer(test_img.tostring(), test_img.shape[1::-1], "RGB")
        
        display.blit(img, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == QUIT:
                capture = False

    pygame.camera.quit()  
