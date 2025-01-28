import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import time

# Load the trained model
def load_model(model_file):
    model = tf.keras.models.load_model(model_file)
    return model

# Dictionary to label emotion categories
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Streamlit title
st.title("Live Emotion Detection")

# Upload the model file (.h5)
model_file = st.file_uploader("Upload Trained Model (.h5 file)", type="h5")

if model_file:
    # Load the uploaded model
    model = load_model(model_file)
    
    # Display message
    st.write("Starting webcam...")

    # OpenCV: Start webcam feed
    cap = cv2.VideoCapture(0)

    # Loop to grab frames from webcam and detect emotions
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load face detection classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract the region of interest (ROI)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=-1)  # add channel dimension
                roi = np.expand_dims(roi, axis=0)  # add batch dimension

                # Make prediction using the model
                prediction = model.predict(roi)[0]
                label = emotion_dict[np.argmax(prediction)]
                label_position = (x, y - 10)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        st.image(frame_rgb, channels="RGB", use_column_width=True)

        # Streamlit updates the image frame
        time.sleep(0.1)  # Control frame rate

    # Release the webcam feed when done
    cap.release()
