import cv2, os, numpy

dataset = ["", "Rajita Ghosal", "Vishal Sinha", "Ashutosh Agarwal", "Aneesh Dixit", "Kshitiz Khatri", "Nihar Chitnis"]

def detect_faces(img) :
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = faceCasc.detectMultiScale(gray, 1.3, 5)
    graylist = []
    faceslist = []

    if len(faces) == 0 :
        return None, None

    for i in range(0, len(faces)) :
        (x, y, w, h) = faces[i]
        graylist.append(gray[y:y+w, x:x+h])
        faceslist.append(faces[i])

    return graylist, faceslist

def detect_face(img) :
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = faceCasc.detectMultiScale(gray, 1.3, 5)
    graylist = []
    faceslist = []

    if len(faces) == 0 :
        return None, None

    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def data() :
    dirs = os.listdir("Dataset")

    faces = []
    labels = []

    for i in dirs :
        set = "Dataset/" + i

        label = int(i)

        for j in os.listdir(set) :
            path = set + "/" + j
            img = cv2.imread(path)
            face, rect = detect_face(img)

            if face is not None :
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels

faces, labels = data()

face_recognizer = cv2.face.createLBPHFaceRecognizer()

face_recognizer.train(faces, numpy.array(labels))

def predict(img) :

    face, rect = detect_faces(img)

    if face is not None :
        for i in range(0, len(face)) :
            label = face_recognizer.predict(face[i])
            label_text = dataset[label]

            return label_text

    return None

video_capture = cv2.VideoCapture(1)

LS = False # Lock system is initially locked

authorized_person = "Rajita Ghosal" # Set the authorized person here

while True :
   
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # If the frame is captured
    if ret:
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Predict the label of the face in the frame
        label_text = predict(frame)

        # If a face is detected
        if label_text is not None:
            # If the detected face belongs to the authorized person
            if label_text == authorized_person:
                # Unlock the system
                LS = True
                # Display "Authorized Personnel" on the screen
                cv2.putText(frame, "Authorized Personnel", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Lock the system
                LS = False
                # Display "Unauthorized Personnel" on the screen
                cv2.putText(frame, "Unauthorized Personnel", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Wait for the 'q' key to be pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video_capture.release()
cv2.destroyAllWindows()