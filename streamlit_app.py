#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit Host Application for Cardiovascular Disease Prediction

This module provides a web interface for predicting cardiovascular disease
using both tabular patient data and medical image analysis.

Author: Ahmad Zahir
Last Modified: April 20, 2025
"""

import os
import sys
import logging
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image, ImageOps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('streamlit_app.log')
    ]
)
logger = logging.getLogger('CardiovascularPredictionApp')

# Check if TensorFlow is available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available and imported successfully")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow is not available. Running in demonstration mode.")

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Get the correct path regardless of where the script is run from
base_dir = os.path.dirname(os.path.abspath(__file__))
tb_model_path = os.path.join(base_dir, "Saved_model", "tb_mdl.h5")
img_model_path = os.path.join(base_dir, "Saved_model", "img_mdl.h5")

# Print paths for debugging
logger.info(f"Base directory: {base_dir}")
logger.info(f"Tabular model path: {tb_model_path}")
logger.info(f"Image model path: {img_model_path}")

# Define input field ranges and descriptions for validation
input_ranges = {
    "age": {"min": 20, "max": 100, "help": "Patient's age in years (20-100)", "unit": "years"},
    "sex": {"min": 0, "max": 1, "help": "Patient's gender (0: female, 1: male)", "unit": ""},
    "cp": {"min": 0, "max": 3, "help": "Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)", "unit": ""},
    "trestbps": {"min": 90, "max": 200, "help": "Resting blood pressure in mm Hg (90-200)", "unit": "mm Hg"},
    "chol": {"min": 120, "max": 570, "help": "Serum cholesterol in mg/dl (120-570)", "unit": "mg/dl"},
    "fbs": {"min": 0, "max": 1, "help": "Fasting blood sugar > 120 mg/dl (0: false, 1: true)", "unit": ""},
    "restecg": {"min": 0, "max": 2, "help": "Resting ECG results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)", "unit": ""},
    "thalach": {"min": 70, "max": 220, "help": "Maximum heart rate achieved (70-220)", "unit": "bpm"},
    "exang": {"min": 0, "max": 1, "help": "Exercise induced angina (0: no, 1: yes)", "unit": ""},
    "oldpeak": {"min": 0, "max": 6.5, "help": "ST depression induced by exercise relative to rest (0-6.5)", "unit": "mm"},
    "slope": {"min": 0, "max": 2, "help": "Slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)", "unit": ""},
    "ca": {"min": 0, "max": 4, "help": "Number of major vessels colored by fluoroscopy (0-4)", "unit": "vessels"},
    "thal": {"min": 0, "max": 3, "help": "Thalassemia (0: normal, 1: fixed defect, 2: reversible defect, 3: unknown)", "unit": ""}
}

def generate_sample_data():
    """
    Generate realistic sample patient data for testing.
    
    Returns:
        dict: Dictionary of sample values for each input field
    """
    try:
        logger.info("Generating sample data for testing")
        sample_data = {}
        sample_data["age"] = random.randint(40, 75)
        sample_data["sex"] = random.randint(0, 1)
        sample_data["cp"] = random.randint(0, 3)
        sample_data["trestbps"] = random.randint(110, 160)
        sample_data["chol"] = random.randint(170, 300)
        sample_data["fbs"] = random.randint(0, 1)
        sample_data["restecg"] = random.randint(0, 2)
        sample_data["thalach"] = random.randint(120, 190)
        sample_data["exang"] = random.randint(0, 1)
        sample_data["oldpeak"] = round(random.uniform(0, 3.5), 1)
        sample_data["slope"] = random.randint(0, 2)
        sample_data["ca"] = random.randint(0, 3)
        sample_data["thal"] = random.randint(0, 2)
        return sample_data
    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}")
        return {}

def validate_input(value, input_type):
    """
    Validate that input is within the expected range.
    
    Args:
        value: The input value to validate
        input_type: The type of input (key in input_ranges)
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if not value:
            return False
            
        if input_type == "oldpeak":
            val = float(value)
        else:
            val = int(float(value))
        
        if val < input_ranges[input_type]["min"] or val > input_ranges[input_type]["max"]:
            return False
        return True
    except ValueError:
        logger.warning(f"Invalid value for {input_type}: {value}")
        return False

@st.cache_resource
def load_tabular_model():
    """
    Load the tabular data prediction model.
    
    Returns:
        model: Loaded TensorFlow model or None if error
    """
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available - can't load model")
        return None
        
    try:
        logger.info(f"Loading tabular model from {tb_model_path}")
        model = tf.keras.models.load_model(tb_model_path)
        logger.info("Tabular model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading tabular model: {str(e)}")
        return None

@st.cache_resource
def load_image_model():
    """
    Load the image-based prediction model.
    
    Returns:
        model: Loaded TensorFlow model or None if error
    """
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available - can't load image model")
        return None
        
    try:
        logger.info(f"Loading image model from {img_model_path}")
        model = tf.keras.models.load_model(img_model_path)
        logger.info("Image model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading image model: {str(e)}")
        return None

def predict_from_tabular_data(inputs, model=None):
    """
    Predict heart disease from tabular patient data.
    
    Args:
        inputs: List of patient feature values
        model: Trained TensorFlow model
        
    Returns:
        tuple: (prediction result, confidence score)
    """
    # If running in demo mode or model not available
    if not TENSORFLOW_AVAILABLE or model is None:
        # Generate a pseudo-random prediction based on the inputs
        # This is just for demonstration when TensorFlow is not available
        logger.info("Running demonstration prediction (TensorFlow not available)")
        
        # Use a deterministic algorithm based on the inputs to simulate consistent predictions
        # Just for demo purposes - not a real medical prediction!
        seed = sum([float(x) for x in inputs])
        random.seed(seed)
        prediction_value = random.uniform(0, 1)
        
        # Make it more likely to predict no disease (for demonstration purposes)
        prediction_value = prediction_value * 0.8
        
        logger.info(f"Demo prediction complete: {prediction_value}")
        
        # Determine result
        if prediction_value > 0.5:
            return ("The person is having heart disease", prediction_value)
        else:
            # For negative predictions, the confidence is (1 - prediction_value)
            # This ensures confidence represents certainty of the prediction
            return ("The person does not have any heart disease", 1.0 - prediction_value)
    
    # Normal prediction with TensorFlow model
    try:
        logger.info("Running prediction on tabular data")
        # Convert inputs to numpy array and reshape
        np_array = np.asarray(inputs).astype('float32')
        reshaped_input = np_array.reshape(1, -1)
        
        # Get prediction
        prediction = model.predict(reshaped_input)
        prediction_value = prediction[0][0]
        
        logger.info(f"Prediction complete: {prediction_value}")
        
        # Determine result
        if prediction_value > 0.5:
            # For positive predictions, use the raw prediction value
            return ("The person is having heart disease", prediction_value)
        else:
            # For negative predictions, the confidence is (1 - prediction_value)
            # This ensures confidence represents certainty of the prediction
            return ("The person does not have any heart disease", 1.0 - prediction_value)
    except Exception as e:
        logger.error(f"Error during tabular prediction: {str(e)}")
        raise

def predict_class(img, model=None):
    """
    Predict heart disease from image data.
    
    Args:
        img: Input image
        model: Trained TensorFlow model
        
    Returns:
        numpy.ndarray: Model prediction
    """
    # If running in demo mode or model not available
    if not TENSORFLOW_AVAILABLE or model is None:
        logger.info("Running demonstration image prediction (TensorFlow not available)")
        
        # Generate a pseudo-random prediction based on the image
        # Just for demonstration when TensorFlow is not available
        img_array = np.array(img)
        avg_pixel = np.mean(img_array)
        
        # Use the average pixel value to generate a deterministic but fake prediction
        # Just for demo purposes - not a real medical prediction!
        random.seed(int(avg_pixel))
        pred_value = random.uniform(0.3, 0.7)
        
        # Create a fake prediction array format matching TensorFlow output
        prediction = np.array([[1 - pred_value, pred_value]])
        logger.info(f"Demo image prediction complete: {prediction}")
        
        return prediction
    
    # Normal prediction with TensorFlow model
    try:
        logger.info("Processing image for prediction")
        # Prepare image
        data = np.ndarray(shape=(1, 299, 299, 3), dtype=np.float32)
        size = (299, 299)
        
        # Resize and normalize image
        image = ImageOps.fit(img, size, Image.LANCZOS)
        image_array = np.asarray(image)
        normalized_image = (image_array.astype(np.float32) / 255.0) - 1
        data[0] = normalized_image
        
        # Run prediction
        prediction = model.predict(data)
        logger.info(f"Image prediction complete: {prediction}")
        
        return prediction
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None

def display_tabular_prediction_page():
    """
    Display the tabular data prediction page.
    """
    # Page title with description
    st.title('Heart Disease Prediction Using Clinical Data')
    st.write("""
    This tool predicts the likelihood of heart disease based on patient clinical data.
    Fill in all the fields below with the patient's information, or use the 'Test with Sample Data' 
    button to try the system with randomly generated values.
    """)
    
    # Load model
    tb_model = load_tabular_model()
    if not TENSORFLOW_AVAILABLE:
        st.warning("⚠️ Running in demonstration mode - TensorFlow is not available on this server. Predictions will be simulated.")
    elif tb_model is None:
        st.error("❌ Failed to load the prediction model. Please check the logs and model file.")
    
    # Initialize session state for form data if it doesn't exist
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {key: "" for key in input_ranges.keys()}
    
    # Form layout
    with st.container():
        # Sample data button with explanatory text
        st.subheader("Quick Test")
        st.write("Click the button below to fill the form with sample data for testing.")
        
        if st.button('Test with Sample Data'):
            st.session_state.form_data = generate_sample_data()
            st.success("✅ Sample data loaded. You can modify any values if needed, then click 'Predict Heart Disease Risk' below.")
            
        # Visual separator
        st.markdown("---")
        st.subheader("Patient Information")
        
        # Form with validation
        input_valid = {}
        col1, col2, col3 = st.columns(3)
        
        # Helper function for consistent input styling
        def create_input(column, field_name, label):
            with column:
                field_info = input_ranges[field_name]
                help_text = f"{field_info['help']}"
                if field_info['unit']:
                    label = f"{label} ({field_info['unit']})"
                
                value = st.text_input(
                    label,
                    value=st.session_state.form_data[field_name],
                    help=help_text
                )
                
                # Validate and show immediate feedback
                is_valid = validate_input(value, field_name) if value else False
                input_valid[field_name] = is_valid
                
                # Show validation status
                if value:
                    if is_valid:
                        st.write("✅ Valid input")
                    else:
                        st.write(f"❌ Invalid input. Valid range: {field_info['min']}-{field_info['max']}")
                
                return value
        
        # Row 1
        age = create_input(col1, "age", "Age")
        sex = create_input(col2, "sex", "Sex")
        cp = create_input(col3, "cp", "Chest Pain Type")
        
        # Row 2
        trestbps = create_input(col1, "trestbps", "Resting Blood Pressure")
        chol = create_input(col2, "chol", "Serum Cholesterol")
        fbs = create_input(col3, "fbs", "Fasting Blood Sugar > 120 mg/dl")
        
        # Row 3
        restecg = create_input(col1, "restecg", "Resting ECG Results")
        thalach = create_input(col2, "thalach", "Maximum Heart Rate")
        exang = create_input(col3, "exang", "Exercise Induced Angina")
        
        # Row 4
        oldpeak = create_input(col1, "oldpeak", "ST Depression")
        slope = create_input(col2, "slope", "Slope of ST Segment")
        ca = create_input(col3, "ca", "Number of Major Vessels")
        
        # Row 5
        thal = create_input(col1, "thal", "Thalassemia")
        
        # Save current form values to session state
        st.session_state.form_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, 
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }
        
        # Display overall validation status
        invalid_inputs = [field for field, valid in input_valid.items() if not valid and st.session_state.form_data[field]]
        if invalid_inputs:
            st.warning(f"⚠️ Invalid values for: {', '.join(invalid_inputs)}. Please check the marked fields.")
        
        # Prediction section
        st.markdown("---")
        st.subheader("Prediction")
        
        # Prediction button with prominent styling
        predict_btn = st.button('Predict Heart Disease Risk', use_container_width=True, type="primary")
        
        if predict_btn:
            # Check if all inputs are valid before prediction
            if all(input_valid.values()):
                try:
                    with st.spinner('Running prediction...'):
                        # Gather inputs
                        inputs = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
                        
                        # Get prediction
                        result, confidence = predict_from_tabular_data(inputs, tb_model)
                        
                        # Display result
                        st.markdown("### Results")
                        
                        # Create columns for result display
                        res_col1, res_col2 = st.columns(2)
                        
                        with res_col1:
                            # Display prediction with appropriate styling
                            if "having heart disease" in result:
                                st.error(f"#### {result}")
                                risk_level = "High"
                            else:
                                st.success(f"#### {result}")
                                risk_level = "Low"
                            
                            st.write(f"Prediction confidence: {confidence*100:.2f}%")
                            st.write(f"Risk level: {risk_level}")
                        
                        with res_col2:
                            # Visualize the prediction confidence
                            fig, ax = plt.subplots(figsize=(4, 2))
                            
                            # Create gauge chart
                            ax.set_xlim(0, 1)
                            ax.set_ylim(0, 1)
                            ax.set_title("Prediction Confidence")
                            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                            ax.set_yticks([])
                            
                            # Add colorbar for risk level
                            cmap = plt.cm.RdYlGn_r
                            norm = plt.Normalize(0, 1)
                            
                            # Draw the gauge bar
                            ax.barh(0.5, 1, height=0.3, color='lightgray', alpha=0.3)
                            ax.barh(0.5, confidence, height=0.3, color=cmap(confidence))
                            
                            # Add a marker for the threshold
                            ax.axvline(0.5, color='gray', linestyle='--')
                            
                            # Display the plot
                            st.pyplot(fig)
                            
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    logger.error(f"Prediction error: {str(e)}")
            else:
                # Identify missing or invalid fields
                missing_fields = [field for field, valid in input_valid.items() if not valid]
                st.error(f"❌ Please fill in all fields with valid values. Check the fields marked with errors.")

def display_image_prediction_page():
    """
    Display the image-based prediction page.
    """
    # Page title with description
    st.title('Heart Disease Prediction Using Medical Images')
    st.write("""
    This tool analyzes heart scan images to predict the likelihood of cardiovascular disease.
    Upload a clear image of a heart scan (jpg or png format) to get a prediction.
    """)
    
    # Show demo mode warning if TensorFlow is not available
    if not TENSORFLOW_AVAILABLE:
        st.warning("⚠️ Running in demonstration mode - TensorFlow is not available on this server. Predictions will be simulated.")
    
    # Load model
    model = load_image_model()
    
    if not TENSORFLOW_AVAILABLE:
        pass  # Already showed warning above
    elif model is None:
        st.error("❌ Failed to load the image model. Please check the logs and model file.")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Upload Image")
        file = st.file_uploader("Select a heart scan image file", type=["jpg", "png"])
        
        if file is None:
            st.info("Waiting for image upload...")
            # Show sample image placeholder
            st.image("https://via.placeholder.com/400x300?text=Sample+Heart+Scan", caption="Sample heart scan image")
        
    with col2:
        if file is not None:
            try:
                # Create a progress indicator
                with st.spinner('Processing image...'):
                    # Load and display the uploaded image
                    test_image = Image.open(file)
                    st.image(test_image, caption="Uploaded Image", width=400)
                    
                    # Predict class
                    pred = predict_class(test_image, model)
                    
                    if pred is not None:
                        # Define class names
                        class_names = ['Negative', 'Positive']
                        
                        # Get prediction result
                        result_index = np.argmax(pred)
                        result = class_names[result_index]
                        confidence = pred[0][result_index]
                        
                        # Display prediction result
                        st.subheader("Prediction Result")
                        
                        # Format result based on prediction
                        if result == 'Positive':
                            st.error(f"#### Cardiovascular Disease Detected")
                            risk_status = "High Risk"
                        else:
                            st.success(f"#### No Cardiovascular Disease Detected")
                            risk_status = "Low Risk"
                            
                        # Show confidence
                        st.write(f"Prediction: {result}")
                        st.write(f"Confidence: {confidence*100:.2f}%")
                        st.write(f"Risk Status: {risk_status}")
                        
                        # Create visualization of confidence
                        st.subheader("Confidence Levels")
                        
                        # Plot confidence levels
                        fig, ax = plt.subplots(figsize=(8, 3))
                        
                        bars = ax.bar(
                            class_names,
                            [pred[0][0], pred[0][1]],
                            color=['green', 'red']
                        )
                        
                        # Add percentage labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(
                                bar.get_x() + bar.get_width()/2.,
                                height,
                                f'{height*100:.1f}%',
                                ha='center',
                                va='bottom'
                            )
                        
                        ax.set_ylim(0, 1.0)
                        ax.set_ylabel('Probability')
                        ax.set_title('Prediction Probabilities')
                        
                        # Display the plot
                        st.pyplot(fig)
                    else:
                        st.error("❌ Failed to process the image. The model could not generate a prediction.")
            except Exception as e:
                st.error(f"❌ Error processing the image: {str(e)}")
                logger.error(f"Image processing error: {str(e)}")

def main():
    """
    Main application function.
    """
    try:
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #424242;
        }
        .info-text {
            font-size: 1rem;
            color: #616161;
        }
        .centered {
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Application header in sidebar
        with st.sidebar:
            st.markdown('<p class="main-header centered">Cardiovascular Disease Prediction System</p>', unsafe_allow_html=True)
            st.markdown('<p class="info-text">Choose a prediction method below</p>', unsafe_allow_html=True)
            
            # Navigation menu
            selected = option_menu(
                menu_title=None,
                options=['Predict with Clinical Data', 'Predict with Heart Scan'],
                icons=['clipboard-data', 'image'],
                default_index=0
            )
            
            # About section in sidebar
            st.markdown("---")
            st.markdown('<p class="sub-header">About</p>', unsafe_allow_html=True)
            st.markdown("""
            This application uses deep learning models to predict cardiovascular disease risk
            using either clinical data or medical images. The models have been trained on validated
            datasets and provide accurate risk assessments.
            
            **Note**: This tool is for educational purposes only and should not replace
            professional medical advice.
            """)
            
            # Display current date and version
            st.markdown("---")
            st.markdown("**Version**: 2.0.0")
            st.markdown("**Last Updated**: April 20, 2025")
            
            # Show environment information
            st.markdown("---")
            st.markdown('<p class="sub-header">Environment</p>', unsafe_allow_html=True)
            st.markdown(f"**Python Version**: {sys.version.split(' ')[0]}")
            st.markdown(f"**TensorFlow Available**: {'Yes' if TENSORFLOW_AVAILABLE else 'No - Running in Demo Mode'}")
            
        # Display the selected page
        if selected == 'Predict with Clinical Data':
            display_tabular_prediction_page()
        else:
            display_image_prediction_page()
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()