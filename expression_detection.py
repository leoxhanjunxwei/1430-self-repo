import numpy as np
import cv2
from keras.models import load_model

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pretrained Mini-Xception for facial expression recognition
model = load_model('test/fer2013_mini_XCEPTION.119-0.65.hdf5')

# Expression labels
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv2.VideoCapture(0)

frame_counter = 0
expression = ""

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_crop = gray[y:y + h, x:x + w]
        face_crop = cv2.resize(face_crop, (48, 48))
        face_crop = face_crop / 255.0
        face_crop = np.expand_dims(face_crop, axis=(0, -1))

        # Predict the expression every 24 frames
        if frame_counter % 24 == 0:
            prediction = model.predict(face_crop)
            max_index = int(np.argmax(prediction))
            expression = labels[max_index]

        # Display the expression
        cv2.putText(frame, expression, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Real-time Face Detection and Expression Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1

# Close webcam and windows
cap.release()
cv2.destroyAllWindows()