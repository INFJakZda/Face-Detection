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

subjects = ["", "John_Travolta", "Julianne_Moore", "Salma_Hayek", "Silvio_Berlusconi", "Zdano", "Reszelo", "huj", "debil"]

def detect_face(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('data/face.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    cord_list = []  #rectangle cordinates list

    if (len(faces) == 0):
        return None, None

    for it in range(len(faces)):
        (x, y, w, h) = faces[it]
        cord_list.append(gray[y:y+w, x:x+h])

    return cord_list, faces

def prepare_training_data(data_folder_path):
    
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    
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
       
        predicted_img = predict(test_img)
        
        img = pygame.image.frombuffer(predicted_img.tostring(), predicted_img.shape[1::-1], "RGB")
        
        display.blit(img, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == QUIT:
                capture = False

    pygame.camera.quit()
