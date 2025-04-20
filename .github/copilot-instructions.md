# Cardiovascular Disease Prediction System

This project implements a deep learning-based prediction system for cardiovascular disease using both tabular data and medical images.

## Directory Structure

```
Cardiovascular-disease-prediction-using-deep-leaerning/
│
├── Application/
│   ├── Code/
│   │   ├── idmcad_(images).py       # Image-based prediction model
│   │   └── idmpcd.py                # Tabular data prediction model
│   ├── dataset/
│   │   └── Tabular_data.csv         # Dataset for tabular model
│   ├── Saved_model/
│   │   ├── img_mdl.h5               # Trained image model
│   │   └── tb_mdl.h5                # Trained tabular data model
│   └── streamlit_host.py            # Streamlit web application
│
├── figures/
│   ├── implementation.png           # Implementation diagram
│   ├── interface.png                # User interface preview
│   └── Use_Case_Diagram.png         # Use case diagram
│
├── Presentation/
│   └── FYP Presentation.pptx        # Project presentation
│
├── Report/
│   └── Ahmad-Zahir-FinalYearProjectDocument.pdf  # Detailed project report
│
├── LICENSE                          # License file
├── README.md                        # Project overview
└── requirements.txt                 # Project dependencies
```

## Setup Instructions

1. Create a virtual environment:
   ```
   uv venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   uv pip install -r requirements.txt
   ```

3. Run the application:
   ```
   cd Application
   streamlit run streamlit_host.py
   ```

## Features

- Predict cardiovascular disease using patient data (tabular form)
- Predict cardiovascular disease using medical images
- User-friendly web interface using Streamlit

## Models

- Tabular data model: Deep neural network trained on clinical features
- Image model: CNN-based model trained on heart scan images

## Technologies Used

- TensorFlow/Keras for deep learning models
- Streamlit for web interface
- NumPy for numerical processing
- Pillow for image processing

# Copilot Instructions

Virtual Environment Setup is already done. You can directly run the application using the commands of windows command prompt or terminal.