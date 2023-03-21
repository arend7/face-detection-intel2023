import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import svc

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def data():
    dirs = os.listdir("Dataset")

    faces = []
    labels = []

    for i in dirs:
        set = "Dataset/" + i

        label = int(i)

        for j in os.listdir(set):
            path = set + "/" + j
            img = cv2.imread(path)
            img = cv2.resize(img, (200, 200))

            face, rect = detect_face(img)

            if face is not None:
                face_resized = cv2.resize(face, (100, 100))
                faces.append(face_resized)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


# Load the face dataset
faces, labels = data()

def recognize_face():
    # Load the dataset
    dirs = os.listdir("Dataset")

    faces = []
    labels = []

    for i in dirs:
        set = "Dataset/" + i

        label = int(i)

        for j in os.listdir(set):
            path = set + "/" + j
            img = cv2.imread(path)
            face, rect = detect_face(img)

            if face is not None:
                faces.append(face)
                labels.append(label)

    # Convert faces to numpy array and apply PCA
    features = np.array([face.flatten() for face in faces])
    pca = PCA(n_components=100)
    features_pca = pca.fit_transform(features)

    # Train the SVM model
    model = svc.SVC(C=1.0, kernel='linear', probability=True)
    model.fit(features_pca, labels)

    # Recognize faces in a new image
    img = cv2.imread("test.jpg")
    face, rect = detect_face(img)
    if face is not None:
        features_test = pca.transform(face.flatten().reshape(1, -1))
        label_test = model.predict(features_test)
        confidence = model.predict_proba(features_test)
        print("Predicted label: {}, Confidence: {}".format(label_test[0], confidence[0][label_test[0]]))

    recognize_face()
# Create a face recognizer

pca = PCA(n_components=100)
features_pca = pca.fit_transform(features)

# Train the face recognizer
features = np.array([face.flatten() for face in faces])
labels = np.array(labels)

features_pca = pca.fit_transform(features)
svc.fit(features_pca, labels)

# Define the video capture device
cap = cv2.VideoCapture(0)

# Loop through the video frames and perform face recognition
while True:
    ret, frame = cap.read()
    face, rect = detect_face(frame)
    if face is not None:
        face_flattened = face.flatten()
        face_pca = pca.transform([face_flattened])
        label = svc.predict(face_pca)[0]
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)
        cv2.putText(frame, str(label), (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device and close the window
cap.release()
cv2.destroyAllWindows()
