#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting module

# Load your trained model
model_path = r'C:\Users\anike\Downloads\YO\Dummy\dataset\fabric_authentication_model_New.h5'
model = tf.keras.models.load_model(model_path)

# Function to preprocess the image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = np.expand_dims(img / 255.0, axis=0)  # Normalize and add batch dimension
    return img

# Function to make predictions and display results with a 3D pie chart
def predict_and_display(image):
    predictions = model.predict(image)
    result_prob = predictions[0][0]

    prediction_label = 'Machine-made' if result_prob > 0.5 else 'Hand-made'
    prediction_probability = result_prob if result_prob > 0.5 else 1 - result_prob

    st.write(f"Prediction: {prediction_label} with probability: {prediction_probability:.4f}")

    # Create a 2D pie chart with improved aesthetics
    fig, ax = plt.subplots()
    labels = [f'{prediction_label} ({prediction_probability:.4f})', f'Other ({1 - prediction_probability:.4f})']
    sizes = [prediction_probability, 1 - prediction_probability]
    explode = (0.1, 0)  # explode 1st slice

    colors = ['#ff9999', '#66b3ff']  # Custom colors
    autopct = lambda p: '{:.1f}%'.format(p) if p > 0 else ''  # Hide percentage for zero values

    ax.pie(sizes, explode=explode, labels=labels, autopct=autopct, startangle=90, shadow=True, colors=colors)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')

    # Set a title
    ax.set_title("Fabric Classification")

    st.pyplot(fig)


# Streamlit app
st.title("Fabric Authentication System")
st.sidebar.title("Options")

# Choose a background color
st.markdown(
    """
    <style>
        .reportview-container {
            background: linear-gradient(to right, #ff6666, #ff8c66, #ffb366, #ffd966, #ffff66, #d9ff66, #b3ff66);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Option to select between upload and webcam
user_choice = st.selectbox("Choose input method:", ("Upload Image", "Capture Webcam"))

if user_choice == "Upload Image":
    # Upload image functionality
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success("Image successfully uploaded!")

        # Button for prediction
        predict_button_upload = st.button("Predict from Uploaded Image")
        if predict_button_upload:
            # Preprocess image and make prediction
            image = np.array(image)
            image = preprocess_image(image)
            predict_and_display(image)

elif user_choice == "Capture Webcam":
    # Display "Predict from Webcam" button only if prediction hasn't been made yet
    predict_button_webcam_clicked = False
    if not predict_button_webcam_clicked:
        predict_button_webcam = st.button("Predict from Webcam")
        if predict_button_webcam:  # Display capture button and prediction upon click
            predict_button_webcam_clicked = True

            # Access webcam
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()

            if ret:
                frame = cv2.flip(frame, 1)  # Flip horizontally for better display
                st.image(frame, channels="BGR", caption="Webcam Image", use_column_width=True)

                # Preprocess image and make prediction
                frame = preprocess_image(frame)
                predict_and_display(frame)

            cap.release()
            cv2.destroyAllWindows()

