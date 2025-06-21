# 🧠 Parkinson’s Disease Prediction

This project uses a machine learning model to predict whether a person has Parkinson’s Disease based on structured medical dataset features — not voice recordings. The goal is to assist in early detection using clinical test results and a Support Vector Machine (SVM) classifier.

---

## 📂 Dataset

- Contains several biomedical features such as:
  - MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)
  - Jitter and Shimmer values
  - HNR, RPDE, DFA, spread measures
- status column:
  - 1 = Parkinson’s patient
  - 0 = Healthy individual
- Dataset is in .csv format and loaded via Google Colab

---

## 🔧 Tools & Libraries Used

- *Python* – Core programming language
- *Pandas* – Reading and analyzing dataset
- *NumPy* – Numerical computations
- *Scikit-learn (sklearn)*:
  - train_test_split – Splitting dataset
  - StandardScaler – Feature normalization
  - SVC (Support Vector Classifier) – Classification model
  - accuracy_score – Model evaluation
- *Google Colab* – For writing and executing the notebook
- *Matplotlib / Seaborn* – (optional) for data visualization

---

## 🚀 Project Workflow

1. Import libraries and load dataset
2. Analyze and preprocess the data
3. Normalize feature values
4. Split dataset into training and testing sets
5. Train the SVM classifier
6. Evaluate model performance using accuracy score

---

## 💻 Code Example

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the data
parkinsons_data = pd.read_csv('/content/Parkinson_disease_cleaned.csv')
