#  Stroke Risk Prediction using Deep Learning

This project presents an end-to-end deep learning solution to **predict the likelihood of stroke** based on a patient’s demographic and health indicators. It was completed as part of a machine learning specialization challenge and simulates a real-world healthcare use case.

---

##  Objective

To assist public health organizations in **identifying individuals most at risk of having a stroke**, using a dataset of patient health records. The goal is to build a **binary classification model** that predicts stroke occurrence, while handling real-world challenges like:

- Imbalanced data
- Noisy or missing values
- Categorical feature encoding
- Choosing the right evaluation metrics for sensitive medical applications

---

##  Dataset Overview

The dataset is sourced from [Kaggle – Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

Each record contains the following features:

| Column Name        | Description                                  |
|--------------------|----------------------------------------------|
| gender             | Gender of the patient                        |
| age                | Age of the patient                           |
| hypertension       | 0 = No, 1 = Has hypertension                 |
| heart_disease      | 0 = No, 1 = Has heart disease                |
| ever_married       | Marital status                               |
| work_type          | Type of employment                           |
| Residence_type     | Urban or rural residence                     |
| avg_glucose_level  | Average glucose level                        |
| bmi                | Body mass index                              |
| smoking_status     | Smoking behavior                             |
| stroke             | 0 = No stroke, 1 = Stroke occurred (Target)  |

---

##  Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras

---

##  Exploratory Data Analysis (EDA)

- Identified missing values in `bmi` and imputed using median.
- Visualized relationships between stroke and:
  - Age
  - Gender
  - Glucose level
  - Smoking status
  - Hypertension
- Detected severe class imbalance (~5% positive stroke cases)

---

##  Modeling Approach

###  Baseline Model
- A basic neural network with two hidden layers (ReLU activation)
- Poor recall and F1-score due to class imbalance

###  Strategy 1: Architecture Improvement
- Added layers and Dropout for regularisation
- Slight improvement in accuracy, but still poor stroke detection

###  Strategy 2: Activation Tuning
- Switched to LeakyReLU to avoid dying neuron issue
- No major performance gain

###  **Strategy 3: Class Weighting (Final Model)**
- Penalized the model more for misclassifying positive stroke cases
- Achieved **Recall = 78%**, **F1 Score = 0.22**
- Selected as **final model** for medical relevance

---

## ✅ Final Evaluation Metrics

| Model                          | Accuracy | Precision | Recall | F1 Score |
|-------------------------------|----------|-----------|--------|----------|
| Baseline                      | 0.95     | 0.33      | 0.02   | 0.03     |
| Strategy 1 – Dropout          | 0.95     | 0.00      | 0.00   | 0.00     |
| Strategy 2 – LeakyReLU        | 0.95     | 0.00      | 0.00   | 0.00     |
| **Strategy 3 – Class Weights**| 0.73     | 0.13      | **0.78**| **0.22**  |

---

## 🩺 Real-World Recommendation

> In stroke prediction, **recall is far more important than raw accuracy**. False negatives (missed stroke cases) are far more dangerous than false positives.  
> 
> The final model (Strategy 3) is suitable for deployment in **clinical decision support systems (CDSS)**, helping healthcare providers flag high-risk patients for further testing or preventive action.

---

##  Project Structure

