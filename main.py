import cv2, os, numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC

dataset = ["","Rajita Ghosal", "Vishal Sinha", "Ashutosh Agarwal", "Aneesh Dixit", "Kshitiz Khatri", "Nihar Chitnis"]

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = faceCasc.detectMultiScale(gray, 1.3, 5)
    graylist = []
    faceslist = []

    if len(faces) == 0:
        return None, None

    for i in range(0, len(faces)):
        (x, y, w, h) = faces[i]
        graylist.append(gray[y:y+w, x:x+h])
        faceslist.append(faces[i])

    return graylist, faceslist

def detect_face(img):q
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = faceCasc.detectMultiScale(gray, 1.3, 5)
    graylist = []
    faceslist = []

    if len(faces) == 0:
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
            face, rect = detect_face(img)

            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels

faces, labels = data()

# Use PCA to extract features from the database
pca = PCA(n_components=100)
features = pca.fit_transform(faces)

# Train an SVM classifier to recognize the person
svm = SVC(kernel='linear', probability=True)
svm.fit(features, labels)

def predict(img):
    face, rect = detect_faces(img)

    if face is not None:
        for i in range(0, len(face)):
            # Use PCA to extract features from the image
            feature = pca.transform(face[i].reshape(1, -1))
            # Use SVM to predict the person in the image
            label = svm.predict(feature)
            label_text = dataset[label[0]]

            (x, y, w, h) = rect[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))
            cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    return img

video_capture = cv2.VideoCapture(1)

while True:
    ret, frame = video_capture.read()

    frame = predict(frame)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
