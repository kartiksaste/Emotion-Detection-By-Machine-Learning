import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io

# Load the trained model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Dictionary to label emotion categories
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Streamlit title
st.title("Emotion Detection from Image")

# Upload the model file
model_file = st.file_uploader("Upload Trained Model (.h5 file)", type="h5")

# Upload image file (only allow .jpeg and .png)
image_file = st.file_uploader("Upload an Image", type=["jpeg", "png"])

if model_file and image_file:
    # Load the uploaded model
    model = load_model(model_file)

    # Load the image for emotion detection
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to an OpenCV format
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.write("No faces found in the image.")
    else:
        # Process the face and predict emotion
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=-1)
                roi = np.expand_dims(roi, axis=0)

                # Make a prediction
                prediction = model.predict(roi, verbose=0)[0]
                label = emotion_dict[np.argmax(prediction)]

                # Draw the face rectangle and label the emotion
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label_position = (x, y - 10)
                cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the image with detected faces and emotions
        st.image(image, caption="Processed Image", use_column_width=True)

else:
    st.write("Please upload both the model file and the image.")
