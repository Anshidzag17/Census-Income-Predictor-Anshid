# 🧠 Income Prediction Using Census Data

Welcome to the **Income Prediction ML Project Assignment**!  

This is a web application that predicts whether a person earns more or less than $50K annually based on features like age, education, capital gain/loss, and marital status. The application uses a trained machine learning model and offers both a Flask-based web interface and a Streamlit interface for interaction.

---

🧠 Model Description:
Model: A classification model (likely Logistic Regression, Random Forest, etc.) stored as classification_model.pkl

Scaler: StandardScaler or similar object saved as scaler.pkl

Target: Predict  income 

📦 Dataset Information

Dataset Name: Adult Census Income

Source: OpenML

Load using:
from sklearn.datasets import fetch_openml

data = fetch_openml("adult", version=2, as_frame=True)

Target column: income

Task type: Binary Classification (>50K or <=50K)

🛠 Features:
* Categorical to numerical feature mapping

* Log-transform on age

* Standardization on selected numeric columns

* Manual mapping for education and sex

Two user interfaces:

 * Flask Web Interface
   
 * Streamlit UI


   
![image](https://github.com/user-attachments/assets/80d207f7-0388-4119-9630-00c83f60db55)



