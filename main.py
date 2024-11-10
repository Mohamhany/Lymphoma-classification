import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
from PIL import Image
import pickle
import joblib

# In your current environment
model = tf.keras.models.load_model('Lymphoma classification.h5')


# Define label mapping
label_map = {
    0: "lymph_cll",
    1: "lymph_fl",
    2: "lymph_mcl"
    }

datagen = ImageDataGenerator()

st.title("Lymphoma Classification")
st.write("Upload an image to classify")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = image.resize((224, 224))  
    image = img_to_array(image)  
    image = np.expand_dims(image, axis=0)  
    image = datagen.flow(image, batch_size=1,).__next__()  

    # Predict using the model
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_label = label_map.get(predicted_class_index)

    # Display prediction
    st.write("Predicted Label:", predicted_label)

