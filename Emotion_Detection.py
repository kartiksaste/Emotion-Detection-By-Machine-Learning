import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Dictionary to label emotion categories
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 
                4: "Neutral", 5: "Sad", 6: "Surprise"}

# Load model only once using session state
if 'model' not in st.session_state:
    st.session_state.model = None

# Initialize video capture in session state
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Streamlit title
st.title("Real-Time Emotion Detection from Webcam")

# Upload model
model_file = st.file_uploader("Upload Trained Model (.h5)", type="h5")
if model_file and not st.session_state.model:
    # Load model into session state
    with st.spinner('Loading model...'):
        st.session_state.model = tf.keras.models.load_model(model_file)

# Webcam control buttons
col1, col2 = st.columns(2)
with col1:
    start_btn = st.button("Start Webcam")
with col2:
    stop_btn = st.button("Stop Webcam")

if start_btn:
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.run = True

if stop_btn and st.session_state.cap:
    st.session_state.cap.release()
    st.session_state.run = False

# Main processing loop
if st.session_state.cap and st.session_state.model and st.session_state.run:
    frame_placeholder = st.empty()
    
    while st.session_state.run:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break

        # Convert to grayscale and detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum(roi_gray) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=(0, -1))
                
                # Predict emotion
                prediction = st.session_state.model.predict(roi, verbose=0)[0]
                label = emotion_dict[np.argmax(prediction)]
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert to RGB and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

    # Cleanup when stopped
    if not st.session_state.run:
        st.session_state.cap.release()
        cv2.destroyAllWindows()
