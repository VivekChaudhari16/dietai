<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:111827,100:ff7e00&height=200&section=header&text=🥗%20NutriPredict%20AI&fontSize=46&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=ML-Powered%20Personalized%20Diet%20Recommendation%20System&descAlignY=58&descSize=18" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-KNN-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-EDA-150458?style=for-the-badge\&logo=pandas\&logoColor=white)](https://pandas.pydata.org)
[![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?style=for-the-badge\&logo=plotly\&logoColor=white)](https://plotly.com)
[![Status](https://img.shields.io/badge/Status-Live-22c55e?style=for-the-badge)](https://diet-vivek.streamlit.app/)

<br/>

### Analyze Health Metrics → Predict Your Diet → Live Better

An end-to-end **Machine Learning-based personalized diet recommendation system** that analyzes health and lifestyle information such as **BMI, blood sugar, cholesterol, activity level, and fitness goals** to recommend a suitable diet category.

<br/>

**[Live Demo](#-live-demo) · [Problem Statement](#-problem-statement) · [ML Workflow](#-ml-workflow) · [Model Evaluation](#-model-evaluation) · [Features](#-input-features) · [Diet Plans](#-predicted-diet-classes) · [Challenges](#-challenges-solved) · [Installation](#-installation-guide) · [Author](#-author)**

</div>

---

# 🥗 NutriPredict AI

## 🚀 Live Demo

<div align="center">

### 🔗 [Open NutriPredict AI](https://diet-vivek.streamlit.app/)

**Enter your health profile → Get an ML-powered diet recommendation**

</div>

---

## 📌 Problem Statement

Generic diet plans do not consider the differences between individuals.

Two people with similar weight or fitness goals may require different recommendations depending on factors such as:

* Age
* BMI
* Activity level
* Blood sugar
* Cholesterol
* Fitness goal

**NutriPredict AI** uses a machine learning approach to analyze these health-related features and predict a suitable diet category.

| Traditional Approach             | NutriPredict AI                        |
| -------------------------------- | -------------------------------------- |
| Generic diet plans               | Personalized recommendations           |
| Manual analysis                  | ML-powered prediction                  |
| Limited health indicators        | BMI, sugar & cholesterol considered    |
| Same recommendation for everyone | Goal and activity-based recommendation |
| Manual calculations              | Automatic BMI and calorie calculations |

---

# 🤖 ML Workflow

```text
                    User Health Input
                           │
                           ▼
              ┌─────────────────────────┐
              │    Data Preprocessing    │
              │                         │
              │ Encoding + BMI +        │
              │ Feature Preparation     │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │       KNN Model         │
              │      model.pkl          │
              │                         │
              │ Finds similar health    │
              │ profiles using distance │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Diet Classification   │
              │                         │
              │ Low Carb                │
              │ Diabetic                │
              │ Heart Healthy           │
              │ Balanced                │
              │ High Protein            │
              └────────────┬────────────┘
                           │
                           ▼
                Personalized Result
                           │
                           ▼
            BMI + Diet + Meal Recommendations
```

---

## 🧠 Why KNN?

The project uses **K-Nearest Neighbors (KNN)** because it is a similarity-based algorithm.

KNN identifies data points that are closest to a user's health profile and uses their neighboring patterns to classify the suitable diet category.

### Advantages

* **Similarity-based:** Useful for matching similar health profiles.
* **Simple and interpretable:** Easy to understand and explain.
* **Non-parametric:** Does not assume a particular data distribution.
* **Suitable for small datasets:** Works effectively for the project's dataset.
* **Easy deployment:** Can be integrated into a Streamlit application.

---

# 📊 Dataset & EDA

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

### Notebooks

| Notebook                                        | Purpose                                                                            |
| ----------------------------------------------- | ---------------------------------------------------------------------------------- |
| `Data Cleaning + EDA.ipynb`                     | Data cleaning, EDA, missing values, outliers and feature analysis                  |
| `Diet Recommendation Using KNN Algorithm.ipynb` | Feature engineering, model training, SMOTE, feature selection and model evaluation |

---

# 📥 Input Features

| # | Feature        | Type        | Description                                     |
| - | -------------- | ----------- | ----------------------------------------------- |
| 1 | Age            | Numeric     | User age                                        |
| 2 | Gender         | Categorical | Gender information                              |
| 3 | Height         | Numeric     | Height in centimetres                           |
| 4 | Weight         | Numeric     | Weight in kilograms                             |
| 5 | BMI            | Derived     | Automatically calculated from height and weight |
| 6 | Activity Level | Categorical | Low, Moderate or High                           |
| 7 | Sugar Level    | Numeric     | Blood sugar reading                             |
| 8 | Cholesterol    | Numeric     | Cholesterol level                               |
| 9 | Goal           | Categorical | Weight Loss, Maintain or Muscle Gain            |

### BMI Calculation

```text
BMI = Weight / Height²
```

where height is converted from centimetres to metres.

---

# 🍽️ Predicted Diet Classes

The model predicts one of the following five diet categories:

| Class | Diet Plan             | General Purpose                                |
| ----- | --------------------- | ---------------------------------------------- |
| 0     | 🥗 Low Carb Diet      | Fat-loss focused profiles                      |
| 1     | 🩺 Diabetic Diet      | Profiles requiring sugar-aware recommendations |
| 2     | ❤️ Heart Healthy Diet | Cholesterol and heart-health focused profiles  |
| 3     | 🍎 Balanced Diet      | General healthy maintenance                    |
| 4     | 💪 High Protein Diet  | Muscle gain and strength-focused profiles      |

### Example Output

```text
Diet Predicted: High Protein Diet

──────────────────────────────────

Breakfast  │ Paneer scramble
Lunch      │ Grilled chicken + rice
Dinner     │ Protein shake + salad
```

> **Note:** These recommendations are generated for the ML project demonstration and should not be treated as medical advice.

---

# 📈 Model Evaluation

Several preprocessing and feature-selection approaches were evaluated before selecting the final KNN configuration.

The evaluation considered both:

* Accuracy
* Macro F1 Score

Macro F1 was considered because the dataset contains class imbalance and accuracy alone may not represent minority-class performance.

## Pipeline Comparison

| # | Pipeline                                      |  Accuracy |
| - | --------------------------------------------- | --------: |
| 1 | Baseline KNN                                  |     73.2% |
| 2 | Forward Feature Selection + SMOTE             |     88.0% |
| 3 | Forward Feature Selection + GridSearchCV      |     86.4% |
| 4 | **Backward Feature Selection + GridSearchCV** | **90.8%** |
| 5 | Correlation-filtered Features + GridSearchCV  |     87.2% |

---

# 🏆 Final Model Configuration

| Parameter                   | Value                                               |
| --------------------------- | --------------------------------------------------- |
| Algorithm                   | K-Nearest Neighbors                                 |
| Feature Selection           | Backward Sequential Feature Selection               |
| Number of Selected Features | 5                                                   |
| Final Features              | BMI, Activity Level, Sugar Level, Cholesterol, Goal |
| Resampling                  | SMOTE                                               |
| Best K                      | **3**                                               |
| Distance Metric             | Manhattan                                           |
| Weight Function             | Distance                                            |
| Hyperparameter Tuning       | GridSearchCV                                        |
| Cross Validation            | Stratified K-Fold                                   |

---

## 📊 Final Performance

```text
Test Accuracy : 90.8%

Macro F1      : 89.4%

Macro Recall  : 92.8%
```

### Per-Class Performance

| Diet Class    | Precision | Recall |    F1 |
| ------------- | --------: | -----: | ----: |
| Balanced Diet |      0.88 |   1.00 |  0.94 |
| Diabetic Diet |      0.98 |   0.88 |  0.93 |
| Heart Healthy |      0.91 |   0.95 |  0.93 |
| High Protein  |      0.72 |   0.87 |  0.79 |
| Low Carb      |      0.85 |  ~0.85 | ~0.85 |

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

# ✨ App Features

### 🎨 Modern UI

Custom Streamlit interface with a clean, responsive design.

### 📊 BMI Gauge

Interactive Plotly gauge showing the calculated BMI.

### 📋 Health KPI Cards

Displays important metrics such as:

* BMI
* Sugar Level
* Cholesterol
* Activity Level

### 🍽️ Meal Recommendations

Provides meal suggestions for:

* Breakfast
* Lunch
* Dinner

### 🔄 Try Again

Users can enter a new health profile without restarting the application.

### 💾 Session State

Streamlit session state is used to maintain the input → prediction → result flow.

---

# 🛠️ Challenges Solved

## 1. Class Imbalance

The dataset contained an uneven distribution between diet categories.

**Solution:**

SMOTE was used to improve representation of minority classes during training.

---

## 2. Feature Selection

Using every available feature did not necessarily improve model performance.

Feature-selection experiments were performed to identify a smaller set of useful features.

The final model uses:

```text
BMI
Activity Level
Sugar Level
Cholesterol
Goal
```

This also makes the final model easier to explain.

---

## 3. Categorical Encoding

Machine learning algorithms require numerical input.

Categorical variables such as:

```text
Gender
Activity Level
Goal
```

were converted into numerical representations before being passed to the KNN model.

---

## 4. Hyperparameter Tuning

Different KNN configurations were evaluated using GridSearchCV.

Parameters such as:

```text
n_neighbors
weights
distance metric
```

were evaluated to identify an effective configuration.

---

## 5. Streamlit State Management

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

# 🧰 Tech Stack

| Layer                 | Technology                 |
| --------------------- | -------------------------- |
| Programming Language  | Python 3.10+               |
| Data Processing       | Pandas, NumPy              |
| Machine Learning      | Scikit-learn               |
| Algorithm             | K-Nearest Neighbors        |
| Imbalanced Data       | SMOTE                      |
| Hyperparameter Tuning | GridSearchCV               |
| Web Application       | Streamlit                  |
| Visualization         | Plotly                     |
| Model Storage         | Pickle                     |
| Development           | Jupyter Notebook / VS Code |
| Deployment            | Streamlit Cloud            |

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
├── Diet Recommendation Using KNN Algorithm.ipynb
├── drs.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation Guide

## 1. Clone the Repository

```bash
git clone https://github.com/Pujadevare445/Diat.git
```

## 2. Navigate to the Project

```bash
cd Diat
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit App

```bash
streamlit run drs.py
```

## 5. Open in Browser

```text
http://localhost:8501
```

> **Note:** The trained model should be available at `model/model.pkl`.

---

# 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

🔗 **https://diet-vivek.streamlit.app/**

---

# 🎯 Project Highlights

* Built an end-to-end ML recommendation system.
* Performed data cleaning and exploratory data analysis.
* Implemented feature engineering.
* Used BMI as a derived health feature.
* Handled class imbalance using SMOTE.
* Performed feature selection.
* Tuned KNN hyperparameters using GridSearchCV.
* Achieved **90.8% test accuracy**.
* Achieved **89.4% Macro F1**.
* Built an interactive Streamlit web application.
* Added Plotly-based BMI visualization.
* Deployed the application online.

---

# 👨‍💻 Author

<div align="center">

### **Vivek Chaudhari**

**Aspiring Data Analyst | Data Scientist | AI/ML Enthusiast**

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-VivekChaudhari16-181717?style=for-the-badge\&logo=github)](https://github.com/VivekChaudhari16)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Vivek%20Chaudhari-0A66C2?style=for-the-badge\&logo=linkedin)](https://www.linkedin.com/in/vivek-chaudhari-9653b1371/)

<br/>

Open to opportunities in **Data Analytics, Data Science, Machine Learning and AI/ML**.

</div>

---

<div align="center">

### 🥗 Built with Python + Machine Learning + Streamlit

**NutriPredict AI**

*Turning health data into personalized diet recommendations.*

⭐ If you find this project useful, consider giving the repository a star!

</div>
