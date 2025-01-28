import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from io import BytesIO
import time

# Load the trained model
def load_model(model_file):
    # Save the uploaded model file temporarily to disk
    with open("temp_model.h5", "wb") as f:
        f.write(model_file.getbuffer())
    
    model = tf.keras.models.load_model("temp_model.h5")
    return model

# Dictionary to label emotion categories
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Streamlit title
st.title("Emotion Detection from Webcam")

# Upload the model file (.h5)
model_file = st.file_uploader("Upload Trained Model (.h5 file)", type="h5")

# Initialize the webcam for live video capture
cap = cv2.VideoCapture(0)

if model_file:
    # Load the uploaded model
    model = load_model(model_file)

    # Loop to capture frames from the webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Region of interest (ROI) for the face
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=-1)
                roi = np.expand_dims(roi, axis=0)

                # Make a prediction
                prediction = model.predict(roi, verbose=0)[0]
                label = emotion_dict[np.argmax(prediction)]
                label_position = (x, y-10)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the image back to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the processed frame
        st.image(frame_rgb, channels="RGB", use_column_width=True)

        # Break the loop if the user presses 'q'
        time.sleep(0.1)

# Release the video capture object
cap.release()
