import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from io import BytesIO

# Load the trained model
def load_model(model_file):
    with open("temp_model.h5", "wb") as f:
        f.write(model_file.getbuffer())
    model = tf.keras.models.load_model("temp_model.h5")
    return model

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

st.title("Real-Time Emotion Detection")

model_file = st.file_uploader("Upload Trained Model (.h5 file)", type="h5")

if model_file:
    model = load_model(model_file)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize session state
    if 'run' not in st.session_state:
        st.session_state.run = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None

    # Start/Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start"):
            st.session_state.run = True
            st.session_state.cap = cv2.VideoCapture(0)
    with col2:
        if st.button("Stop"):
            st.session_state.run = False
            if st.session_state.cap:
                st.session_state.cap.release()

    frame_placeholder = st.empty()

    # Process frames if webcam is running
    if st.session_state.run and st.session_state.cap.isOpened():
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to capture frame")
            st.session_state.run = False
        else:
            # Convert to grayscale and detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum(roi_gray) != 0:
                    roi = roi_gray.astype('float')/255.0
                    roi = np.expand_dims(roi, axis=-1)
                    roi = np.expand_dims(roi, axis=0)
                    prediction = model.predict(roi, verbose=0)[0]
                    label = emotion_dict[np.argmax(prediction)]
                    cv2.putText(frame, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")
            
            # Rerun to update the frame
            st.experimental_rerun()
