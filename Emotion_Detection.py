import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from io import BytesIO

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

if model_file:
    # Load the uploaded model
    model = load_model(model_file)

    # Start webcam capture using Streamlit's built-in camera_input widget
    st.write("Starting webcam...")

    # Capture the image from the webcam
    frame = st.camera_input("Take a photo")

    if frame is not None:
        # Convert the image to an array
        image = np.array(bytearray(frame), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
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
                cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(image, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the image back to RGB for Streamlit display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the processed image with emotion prediction
        st.image(image_rgb, channels="RGB", use_column_width=True)
