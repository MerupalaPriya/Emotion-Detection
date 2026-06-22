import cv2
import numpy as np
from keras.models import load_model

# Load model without compiling (avoids optimizer error)
model = load_model('emotion_model.h5', compile=False)

# Emotion labels (adjust if your model has different order)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))          # Resize to 64x64
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=-1)                 # Add channel dimension: (64,64,1)
        roi = np.expand_dims(roi, axis=0)                  # Add batch dimension: (1,64,64,1)

        # Predict emotion
        predictions = model.predict(roi, verbose=0)
        emotion = emotion_labels[int(np.argmax(predictions))]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)

    cv2.imshow('Emotion Detector', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
