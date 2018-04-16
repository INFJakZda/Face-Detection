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
        
        test_img = cv2.cvtColor(np.array(test_img), cv2.COLOR_BGR2RGB)
        
        #img = pygame.image.frombuffer(predicted_img.tostring(), predicted_img.shape[1::-1], "RGB")
        img = pygame.image.frombuffer(test_img.tostring(), test_img.shape[1::-1], "RGB")
        
        display.blit(img, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == QUIT:
                capture = False

    pygame.camera.quit()  
