import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---------------------------
# 1. Load pre-trained model
# ---------------------------
# Replace with your model file path if different
model = load_model('emotion_model.h5')

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Labels (make sure these match your trained model)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

img_size = (48, 48)

# ---------------------------
# 2. Open webcam and predict
# ---------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, img_size)
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        preds = model.predict(roi)
        label = emotion_labels[np.argmax(preds)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
