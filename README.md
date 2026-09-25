<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:111827,100:ff7e00&height=200&section=header&text=🥗%20NutriPredict%20AI&fontSize=46&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=ML-Powered%20Personalized%20Diet%20Recommendation%20System&descAlignY=58&descSize=18" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-EDA-150458?style=for-the-badge\&logo=pandas\&logoColor=white)](https://pandas.pydata.org)
[![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=for-the-badge\&logo=plotly\&logoColor=white)](https://plotly.com)
[![Status](https://img.shields.io/badge/Status-Live-22c55e?style=for-the-badge)](https://diet-vivek.streamlit.app/)

<br/>

### Analyze Health Metrics → Predict Your Diet → Live Better

An end-to-end **Machine Learning-based personalized diet recommendation system** that analyzes health and lifestyle information such as **BMI, blood sugar, cholesterol, activity level, and fitness goals** to recommend a suitable diet category.

<br/>

**[Live Demo](#-live-demo) · [Problem Statement](#-problem-statement) · [ML Workflow](#-ml-workflow) · [Models](#-machine-learning-models) · [Model Comparison](#-model-comparison) · [Features](#-input-features) · [Diet Plans](#-predicted-diet-classes) · [Challenges](#-challenges-solved) · [Installation](#-installation-guide) · [Author](#-author)**

</div>

---

# 🥗 NutriPredict AI

## 🚀 Live Demo

<div align="center">

### 🔗 [Open NutriPredict AI](https://diet-vivek.streamlit.app/)

**Enter your health profile → Get an ML-powered diet recommendation**

</div>

---

# 📌 Problem Statement

Generic diet plans do not consider the differences between individuals.

Two people with similar weight or fitness goals may have different health and lifestyle characteristics such as:

* Age
* BMI
* Activity level
* Blood sugar
* Cholesterol
* Fitness goal
* Dietary preference
* Region
* Budget

**NutriPredict AI** uses machine learning to analyze these inputs and classify a user into a suitable diet category.

| Traditional Approach             | NutriPredict AI                        |
| -------------------------------- | -------------------------------------- |
| Generic diet plans               | Personalized recommendations           |
| Manual analysis                  | ML-powered prediction                  |
| Limited health indicators        | BMI, sugar & cholesterol considered    |
| Same recommendation for everyone | Goal and activity-based recommendation |
| Manual calculations              | Automatic BMI and calorie calculations |
| Static recommendations           | Interactive web application            |

---

# 🤖 ML Workflow

```text
                    User Health Input
                           │
                           ▼
              ┌─────────────────────────┐
              │   Data Cleaning & EDA   │
              │                         │
              │ Missing Values          │
              │ Duplicates              │
              │ Outliers                │
              │ Data Analysis           │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Feature Engineering   │
              │                         │
              │ BMI Calculation         │
              │ Encoding                │
              │ Feature Preparation     │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │    Class Balancing      │
              │                         │
              │          SMOTE          │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │    Model Training       │
              │                         │
              │ • Random Forest         │
              │ • Gradient Boosting     │
              │ • XGBoost               │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Model Evaluation      │
              │                         │
              │ Accuracy                │
              │ Precision               │
              │ Recall                  │
              │ F1 Score                │
              └────────────┬────────────┘
                           │
                           ▼
                  Diet Classification
                           │
                           ▼
              Personalized Recommendation
                           │
                           ▼
                 BMI + Diet + Meals
```

---

# 🧠 Machine Learning Models

NutriPredict AI explores multiple machine learning approaches for diet classification.

## 🌳 1. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make a final prediction.

It was used to identify nonlinear relationships between health and lifestyle features.

### Key Advantages

* Handles nonlinear relationships
* Reduces overfitting compared with a single decision tree
* Works well with structured/tabular data
* Can provide feature importance
* Suitable for classification problems

---

## 📈 2. Gradient Boosting

Gradient Boosting is an ensemble learning technique that builds models sequentially.

Each new model attempts to correct the errors made by previous models.

```text
Initial Model
      ↓
Calculate Errors
      ↓
New Model Learns From Errors
      ↓
Repeat
      ↓
Final Prediction
```

### Key Advantages

* Captures nonlinear relationships
* Sequentially improves predictions
* Works well with structured data
* Can provide strong classification performance

---

## 🚀 3. XGBoost

XGBoost stands for **Extreme Gradient Boosting**.

It is an optimized gradient boosting algorithm based mainly on decision trees.

The model builds trees sequentially and uses regularization and gradient-based optimization to improve predictive performance.

### Key Advantages

* Strong performance on tabular data
* Handles nonlinear relationships
* Supports regularization
* Efficient training
* Supports hyperparameter tuning
* Provides feature importance

---

# ⚙️ Gradient Descent

Gradient Descent is an **optimization algorithm**, not a classification model.

It is used to minimize a loss or cost function by updating model parameters iteratively.

The basic update rule is:

```text
θnew = θold - learning_rate × gradient
```

### Main Concepts

* Loss function
* Gradient
* Learning rate
* Parameter updates
* Iterative optimization

Gradient Descent is an important optimization concept in machine learning and is used as the foundation for optimization in many predictive models.

---

# 🔬 Model Comparison

Multiple machine learning models were explored during the development of NutriPredict AI.

| Model             | Type                   | Purpose                |
| ----------------- | ---------------------- | ---------------------- |
| Random Forest     | Ensemble Tree Model    | Classification         |
| Gradient Boosting | Boosting Ensemble      | Classification         |
| XGBoost           | Optimized Boosting     | Classification         |
| Gradient Descent  | Optimization Algorithm | Parameter Optimization |

The models were evaluated using classification metrics to understand their performance on the diet recommendation problem.

### Model Performance

| Model             |        Accuracy |       Precision |          Recall |        F1 Score |
| ----------------- | --------------: | --------------: | --------------: | --------------: |
| Random Forest     | **[Add Score]** | **[Add Score]** | **[Add Score]** | **[Add Score]** |
| Gradient Boosting | **[Add Score]** | **[Add Score]** | **[Add Score]** | **[Add Score]** |
| XGBoost           | **[Add Score]** | **[Add Score]** | **[Add Score]** | **[Add Score]** |

> Replace the values above with the actual results from your notebook. No model performance numbers should be added unless they come from the actual evaluation.

---

# 📊 Dataset & Exploratory Data Analysis

The project contains approximately **1,000 health records** covering different user profiles and diet categories.

### Data preprocessing included:

* Missing-value analysis
* Duplicate checking
* Data type inspection
* Outlier analysis
* Correlation analysis
* Feature engineering
* Categorical encoding
* Class distribution analysis
* Feature selection
* Data balancing using SMOTE

### Notebooks

| Notebook                             | Purpose                                                           |
| ------------------------------------ | ----------------------------------------------------------------- |
| `Data Cleaning + EDA.ipynb`          | Data cleaning, EDA, missing values, outliers and feature analysis |
| `Diet Recommendation Using ML.ipynb` | Model training, feature engineering, SMOTE and model evaluation   |

---

# 📥 Input Features

The application accepts health and lifestyle information from the user.

| #  | Feature         | Type                | Description                          |
| -- | --------------- | ------------------- | ------------------------------------ |
| 1  | Age             | Numeric             | User age                             |
| 2  | Gender          | Categorical         | Gender information                   |
| 3  | Height          | Numeric             | Height in centimetres                |
| 4  | Weight          | Numeric             | Weight in kilograms                  |
| 5  | BMI             | Derived             | Calculated from height and weight    |
| 6  | Activity Level  | Categorical         | User's activity level                |
| 7  | Sugar Level     | Numeric             | Blood sugar reading                  |
| 8  | Cholesterol     | Numeric             | Cholesterol level                    |
| 9  | Goal            | Categorical         | Weight Loss, Maintain or Muscle Gain |
| 10 | Diet Preference | Categorical         | Vegetarian / Non-Vegetarian          |
| 11 | Region          | Categorical         | North / South Indian                 |
| 12 | Budget          | Numeric/Categorical | User's food budget                   |

---

# 🧮 BMI Calculation

BMI is calculated automatically using height and weight.

```text
BMI = Weight (kg) / Height² (m²)
```

For example:

```text
Weight = 70 kg
Height = 1.75 m

BMI = 70 / (1.75 × 1.75)
    = 22.86
```

The calculated BMI is then used as an important health-related feature.

---

# 🍽️ Predicted Diet Classes

The system predicts one of the following diet categories:

| Class | Diet Plan             | General Purpose                               |
| ----- | --------------------- | --------------------------------------------- |
| 0     | 🥗 Low Carb Diet      | Fat-loss focused profiles                     |
| 1     | 🩺 Diabetic Diet      | Sugar-aware recommendations                   |
| 2     | ❤️ Heart Healthy Diet | Cholesterol and heart-health focused profiles |
| 3     | 🍎 Balanced Diet      | General healthy maintenance                   |
| 4     | 💪 High Protein Diet  | Muscle gain and strength-focused profiles     |

### Example Output

```text
Diet Predicted: High Protein Diet

──────────────────────────────────

Breakfast  │ Paneer scramble
Lunch      │ Grilled chicken + rice
Dinner     │ Protein shake + salad
```

> **Note:** These recommendations are generated for an ML project demonstration and should not be treated as medical advice.

---

# 📈 Model Evaluation

The models were evaluated using multiple classification metrics.

### Evaluation Metrics

**Accuracy**

Measures the percentage of correctly classified predictions.

**Precision**

Measures how many predicted instances of a class were actually correct.

**Recall**

Measures how many actual instances of a class were correctly identified.

**F1 Score**

Provides a balance between precision and recall.

```text
F1 Score = 2 × (Precision × Recall)
           ──────────────────────────
             Precision + Recall
```

Macro-averaged metrics can also be used when the dataset contains class imbalance because they give equal importance to each class.

---

# ⚖️ Handling Class Imbalance

The dataset contained an uneven distribution among different diet categories.

To address this issue, **SMOTE (Synthetic Minority Over-sampling Technique)** was used during model development.

```text
Original Dataset
       ↓
Identify Minority Classes
       ↓
Generate Synthetic Samples
       ↓
Balanced Training Data
       ↓
Model Training
```

SMOTE was applied to the training data to reduce the impact of class imbalance.

---

# 🎯 Feature Engineering

Feature engineering was performed to create useful information from the raw inputs.

### BMI

Height and weight were used to calculate BMI.

```text
BMI = Weight / Height²
```

### Categorical Encoding

Categorical features such as:

```text
Gender
Activity Level
Goal
Diet Preference
Region
```

were converted into numerical representations before model training.

---

# 🔍 Feature Selection

Feature selection was performed to identify useful variables for model training.

Important health-related features included:

```text
BMI
Activity Level
Sugar Level
Cholesterol
Goal
```

Feature selection helps:

* Reduce unnecessary features
* Simplify the model
* Improve interpretability
* Reduce computational complexity
* Focus on relevant information

---

# 🧩 Challenges Solved

## 1. Class Imbalance

### Problem

Some diet categories had fewer samples than others.

### Solution

SMOTE was used on the training data to generate synthetic samples for minority classes.

---

## 2. Feature Selection

### Problem

Using every available feature does not always improve model performance.

### Solution

Feature-selection techniques were explored to identify relevant health and lifestyle features.

---

## 3. Categorical Data

### Problem

Machine learning algorithms require numerical inputs.

### Solution

Categorical variables were encoded into numerical representations before model training.

---

## 4. Model Selection

### Problem

Different machine learning algorithms can behave differently on structured health datasets.

### Solution

Multiple models were explored:

```text
Random Forest
Gradient Boosting
XGBoost
```

Their classification performance was evaluated using multiple metrics.

---

## 5. Hyperparameter Tuning

Model parameters were tuned to improve classification performance.

Parameters explored included model-specific settings such as:

```text
Number of Estimators
Tree Depth
Learning Rate
Minimum Samples
Regularization Parameters
```

The final configuration depends on the model selected for deployment.

---

## 6. Streamlit State Management

Streamlit reruns the Python script whenever a widget changes.

`st.session_state` was therefore used to maintain the application flow between:

```text
Input Screen
      ↓
Prediction
      ↓
Results Dashboard
```

---

# ✨ Application Features

## 🎨 Modern UI

A custom Streamlit interface provides a clean and responsive user experience.

## 📊 BMI Gauge

An interactive Plotly gauge displays the calculated BMI.

## 📋 Health KPI Cards

The application displays important metrics such as:

* BMI
* Sugar Level
* Cholesterol
* Activity Level

## 🍽️ Meal Recommendations

The application provides meal suggestions for:

* Breakfast
* Lunch
* Dinner

## 🔄 Try Again

Users can enter a new health profile and generate another recommendation.

## 💾 Session State

Streamlit session state maintains the input → prediction → result workflow.

---

# 🛠️ Tech Stack

| Layer                 | Technology                                |
| --------------------- | ----------------------------------------- |
| Programming Language  | Python 3.10+                              |
| Data Processing       | Pandas, NumPy                             |
| Machine Learning      | Scikit-learn                              |
| Models                | Random Forest, Gradient Boosting, XGBoost |
| Optimization          | Gradient Descent                          |
| Imbalanced Data       | SMOTE                                     |
| Hyperparameter Tuning | GridSearchCV                              |
| Feature Selection     | Feature Selection Techniques              |
| Web Application       | Streamlit                                 |
| Visualization         | Plotly                                    |
| Model Storage         | Pickle                                    |
| Development           | Jupyter Notebook / VS Code                |
| Deployment            | Streamlit Community Cloud                 |

---

# 📸 App Preview

<p align="center">

<img src="assets/Project Screenshots/desktop view.png" width="48%" alt="NutriPredict AI Input Screen" />

<img src="assets/Project Screenshots/desktop-results 1.png" width="48%" alt="NutriPredict AI Results Dashboard" />

</p>

<p align="center">

<i>NutriPredict AI input interface and personalized diet recommendation dashboard.</i>

</p>

---

# 📂 Project Structure

```text
NutriPredict-AI/

│
├── .streamlit/
│   └── config.toml
│
├── assets/
│   ├── Project Screenshots/
│   │   ├── desktop view.png
│   │   ├── desktop-results 1.png
│   │   ├── desktop-results 2.png
│   │   ├── mobile view.jpeg
│   │   ├── mobile-results 1.jpeg
│   │   └── mobile-results 2.jpeg
│   │
│   └── images/
│       └── bg1.png
│
├── model/
│   └── model.pkl
│
├── Data Cleaning + EDA.ipynb
├── Diet Recommendation Using ML.ipynb
├── drs.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation Guide

## 1. Clone the Repository

```bash
git clone https://github.com/VivekChaudhari16/Diat.git
```

## 2. Navigate to the Project

```bash
cd Diat
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit Application

```bash
streamlit run drs.py
```

## 5. Open in Browser

```text
http://localhost:8501
```

> **Note:** Make sure the trained model is available at `model/model.pkl`.

---

# 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

### 🔗 Live Application

**https://diet-vivek.streamlit.app/**

---

# 🎯 Project Highlights

* Built an end-to-end **Machine Learning-based diet recommendation system**.
* Performed data cleaning and exploratory data analysis.
* Implemented feature engineering.
* Calculated BMI automatically from height and weight.
* Encoded categorical variables for machine learning.
* Handled class imbalance using **SMOTE**.
* Performed feature selection.
* Experimented with **Random Forest, Gradient Boosting and XGBoost**.
* Used **Gradient Descent as an optimization concept**.
* Evaluated models using Accuracy, Precision, Recall and F1 Score.
* Implemented hyperparameter tuning.
* Built an interactive Streamlit web application.
* Added Plotly-based BMI visualization.
* Added personalized meal recommendations.
* Implemented Streamlit session state.
* Deployed the application online.

---

# 👨‍💻 Author

<div align="center">

### **Vivek Chaudhari**

**Aspiring Data Analyst | Data Scientist | AI/ML Enthusiast**

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-VivekChaudhari16-181717?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/VivekChaudhari16)
