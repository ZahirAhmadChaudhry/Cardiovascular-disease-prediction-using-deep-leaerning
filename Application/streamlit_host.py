# -*- coding: utf-8 -*-

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from streamlit_option_menu import option_menu
from PIL import Image, ImageOps

tb_model = tf.keras.models.load_model(r"C:/FYP/Application/Saved_model/tb_mdl.h5")

# Sidbar for Navigation

with st.sidebar:
    selected = option_menu('Coronary Artery Disease Prediction System',
                           
                           ['Predit by Filling Up Form',
                            'Predict Using Images'],
                           
                           icons = ['activity','heart'],
                           menu_icon="award", 
                           
                           default_index = 0)

#Page for Tabular Data
if (selected == 'Predit by Filling Up Form'):
    
    # page title
    st.title('Heart Disease Prediction Using Deep Learning')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        inputs = (age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal)
        npArray = np.asarray(inputs).astype('float32')
        inReshaped = npArray.reshape(1,-1)
        heart_prediction = tb_model.predict(inReshaped)                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)

def load_model():
        model = tf.keras.models.load_model('C:/FYP/Application/Saved_model/img_mdl.h5')
        return model


def predict_class(img, model):

        image = img
        data = np.ndarray(shape=(1,299,299,3), dtype=np.float32)
        size = (299, 299)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        Nr_image_array = (image_array.astype(np.float32) / 255.0) -1

        data[0] = Nr_image_array

        prediction = model.predict(data)

        return prediction

if (selected == 'Predict Using Images'):  
    #page title
    st.title('Heart Disease Prediction Using Deep Learning')
    model = load_model()
    file = st.file_uploader("Upload an image of a flower", type=["jpg", "png"])
    if file is None:
        st.text('Waiting for upload....')
    else:
        slot = st.empty()
        slot.text('Running inference....')

        test_image = Image.open(file)

        st.image(test_image, caption="Input Image", width = 400)

        pred = predict_class(test_image, model)
        print("This is test", pred)

        class_names = ['Negative', 'Positive']

        result = class_names[np.argmax(pred)]

        output = 'The image is a ' + result

        slot.text('Done')

        st.success(output)