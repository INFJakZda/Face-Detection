import cv2
import os
import subprocess
import time
import numpy as np

subjects = ["", "John_Travolta", "Julianne_Moore", "Salma_Hayek", "Silvio_Berlusconi", "Zdano"]

def detect_face(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('data/face.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

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
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
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
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    img = test_img.copy()
    faces, rect = detect_face(img)
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

    test_img1 = cv2.imread("pictures/salma-travolta-hayek.png")
    
    print("Predicting images...")

    predicted_img1 = predict(test_img1)
    print("Prediction complete")

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", predicted_img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
