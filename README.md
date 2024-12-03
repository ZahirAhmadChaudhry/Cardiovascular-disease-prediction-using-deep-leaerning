 # Cardiovascular Disease Prediction Project

This project aims to develop a deep learning model for predicting cardiovascular disease (CVD) by utilizing artificial neural networks (ANN) and convolutional neural networks (CNN) to analyze both structured and unstructured CVD-related data.

### 1. Cardiovascular Disease Prediction Project Interface
This interface demonstrates how users interact with the system. Key features include:
- Intuitive data input fields for user convenience.
- An output section displaying risk predictions clearly.
- Back-end integration of machine learning algorithms for prediction analysis.

![Cardiovascular Disease Prediction Project Interface](figures/interface.png)

---

### 2. Use Case Diagram
The use case diagram highlights user-system interactions:

![Use Case Diagram](figures/Use_Case_Diagram.png)

---

### 3. Project Implementation Workflow
This workflow illustrates the implementation process:
- **Data Collection**
- **Preprocessing**
- **Model Training**
- **Deployment**

![Project Implementation](figures/implementation.png)


### Research Objectives

This project focuses on addressing two key research objectives related to cardiovascular disease (CVD) prediction.

#### **1. Can we accurately predict the presence of cardiovascular disease based on given features?**

The primary goal of this research is to determine if machine learning models can effectively identify whether a person has cardiovascular disease using various health-related features. These features may include:

- **Demographic Information**: Age, gender, etc.
- **Lifestyle Factors**: Smoking status, physical activity levels, and alcohol consumption.
- **Medical History**: Information about pre-existing conditions like hypertension, diabetes, or previous heart-related issues.
- **Clinical Measurements**: Blood pressure, cholesterol levels, body mass index (BMI), etc.

The focus is on building and validating a predictive model that can take these inputs and classify a person as either "having CVD" or "not having CVD" with a high degree of accuracy. By comparing multiple algorithms such as logistic regression, decision trees, random forests, and neural networks, this research aims to identify the most reliable and interpretable model for CVD prediction.

Key steps in achieving this objective:
1. **Data Collection**: Obtain a dataset with comprehensive and labeled CVD-related records.
2. **Preprocessing**: Handle missing data, normalize values, and remove outliers to ensure the quality of the inputs.
3. **Feature Selection**: Analyze which variables are most relevant for prediction.
4. **Model Evaluation**: Test different algorithms and use metrics like accuracy, precision, recall, and F1-score to evaluate their performance.

---

#### **2. What are the most important features contributing towards CVD prediction?**

The second objective is to identify the specific features that have the most significant impact on the prediction of cardiovascular disease. This involves understanding which variables are consistently associated with an increased or decreased risk of CVD. For instance:

- **Are high cholesterol levels more influential than high blood pressure?**
- **How does age contribute to the risk compared to lifestyle factors like smoking?**

Understanding feature importance not only improves the model's interpretability but also offers valuable knowledge for medical practitioners to focus on the most critical risk factors during diagnosis and treatment.

---

### Significance of These Objectives
- **For Healthcare**: Accurate prediction models and feature importance insights can guide doctors in early diagnosis and personalized treatment plans.
- **For Preventive Measures**: Public health initiatives can target the most critical risk factors identified in the study.
- **For Future Research**: Insights gained can contribute to the development of more advanced, domain-specific models or datasets for cardiovascular disease studies. 

By addressing these objectives, the project seeks to provide actionable, data-driven tools and insights for combating cardiovascular diseases more effectively.

## 📁 Project Structure

- `data` : Contains the datasets utilized throughout the project.
	+ `UCI ML Repository`
	+ `Mendeley Data`
- `docs` : Documentation files (e.g. PDF reports).
- `notebooks` : Jupyter Notebooks for exploratory data analysis, modeling, and visualization.
- `src` : Source code implementation (e.g. custom layers, optimizer classes, etc.).
- `presentation` : Slides summarizing key findings.
- `project_diagrams` : Architecture diagrams and flowcharts.



## Tech Stack

🚀 **Languages**
- Python

📊 **Data Analysis and Visualization**:
- Pandas
- Matplotlib
- Plotly
- Seaborn
- Bokeh

🤖 **Machine Learning and Deep Learning**:
- TensorFlow
- Keras
- PyTorch Lightning
- FastAI
- scikit-learn
- numpy
- tqdm

📚 **Other Tools and Frameworks**:
- ipynb: Jupyter Notebook format for reproducible research workflows
- nbconvert: Convert .ipynb files into multiple formats like HTML, LaTeX, PDF, Markdown, etc.

🔧 **Version Control**:
- Git: Collaborative version control system

💻 **Development Environment**:
- Google Colab: Free GPU-enabled environment supporting popular libraries out-of-box
- Visual Studio Code: Advanced text editor offering numerous extensions catered specifically towards ML/DL projects

## 🔧 Deployment

To deploy this project locally, follow these steps:

1. Clone the repository onto your local machine:
    ```bash
      git clone <https://https://github.com/ZahirAhmadChaudhry/Cardiovascular-disease-prediction-using-deep-leaerning>
    ```
2. Install all dependencies specified in the `requirements.txt` file:
    ```bash
      pip install -r requirements.txt
    ```
3. Optionally create a virtual environment before running any commands if desired:
   ```bash
     conda env create --name cvd_env python==3.9
     conda activate cvd_env
   ```

## 📰 Project Report

For more comprehensive details on the study, results, and future recommendations, review the [Project Report](https://github.com/ZahirAhmadChaudhry/Cardiovascular-disease-prediction-using-deep-leaerning/blob/main/Report/Ahmad-Zahir-FinalYearProjectDocument.pdf).

## 📄 License

This project is distributed under the terms set forth by the [MIT License](https://github.com/ZahirAhmadChaudhry/Cardiovascular-disease-prediction-using-deep-leaerning/blob/main/LICENSE).

## 🌐 Additional Resources

- [NIH - National Heart, Lung, and Blood Institute](<https://www.nhlbi.nih.gov/>): General overview of heart health and various conditions affecting it.
- [CDC - Centers for Disease Control and Prevention](<https://www.cdc.gov/heartdisease/index.htm>): Statistics, causes, risk factors, prevention tips, and resources regarding heart disease.
- [Mayo Clinic - Cardiovascular Disease Definition](<https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118>): Comprehensive definition and explanation of cardiovascular diseases.