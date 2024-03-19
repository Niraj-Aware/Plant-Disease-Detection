# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Loading the Model (Replace './dataset/disease.h5' with the path to your disease model)
model = load_model('./dataset/disease.h5')

# Name of Classes and their details
CLASS_DETAILS = {
    'Corn-Common_rust': {
        'details': 'Common rust is a fungal disease that affects the leaves of corn plants.',
        'recommendations': [
            'Apply fungicides containing active ingredients like azoxystrobin or propiconazole.',
            'Plant resistant corn varieties if available.',
            'Remove and destroy crop residues after harvest to reduce overwintering of the pathogen.'
        ]
    },
    'Potato-Early_blight': {
        'details': 'Early blight is a common fungal disease affecting potato plants.',
        'recommendations': [
            'Practice crop rotation to reduce disease buildup in the soil.',
            'Apply fungicides containing active ingredients like chlorothalonil or mancozeb.',
            'Avoid overhead irrigation to minimize leaf wetness.'
        ]
    },
    'Tomato-Bacterial_spot': {
        'details': 'Bacterial spot is a common disease affecting tomato plants.',
        'recommendations': [
            'Use disease-free seeds or transplants from reputable sources.',
            'Prune infected plant parts and destroy them to reduce disease spread.',
            'Apply copper-based fungicides to control bacterial spot.'
        ]
    }
}

# Setting Title of App
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

# Uploading the plant leaf image
plant_image = st.file_uploader("Choose an image...", type="jpg")

# On predict button click
if st.button('Predict'):
    if plant_image is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)

        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256, 256))

        # Convert image to 4 Dimension
        opencv_image = np.expand_dims(opencv_image, axis=0)

        # Make Prediction
        predictions = model.predict(opencv_image)
        predicted_class = np.argmax(predictions)
        disease_name = list(CLASS_DETAILS.keys())[predicted_class]
        disease_details = CLASS_DETAILS[disease_name]['details']
        disease_recommendations
