import streamlit as st 
import pandas as pd
import numpy as np
import pickle

with open("classification_model.pkl","rb") as model_file:
    model=pickle.load(model_file)
scaler=pickle.load(open("scaler.pkl",'rb'))

st.title("Income Prediction App")

#input fields for the user
education=st.selectbox("education",['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th',
                                     'HS-grad','Some-college','Assoc-voc','Assoc-acdm','Bachelors',
                                     'Masters','Prof-school','Doctorate'])
sex=st.selectbox("sex",["Male","Female"])
age=st.number_input("age")
capital_gain=st.number_input("capital-gain")
capital_loss=st.number_input("capital-loss")
hours_per_week=st.number_input("hours-per-week")
marital_Never_married=st.number_input("marital_Never-married")

#creating a dataframe with user inputs and applying one hot encoding
input_data=pd.DataFrame({
    'age':[age],
    'education':[education],
    'sex':[sex],
    'capital-gain':[capital_gain],
    'capital-loss':[capital_loss],
    'hours-per-week':[hours_per_week],
    'marital_Never-married':[marital_Never_married]       
})

#mapping categorical variables manually
input_data['sex']=input_data['sex'].map({"Male":0,"Female":1})
input_data['education']=input_data['education'].map({'Preschool':0,'1st-4th':1,'5th-6th':2,'7th-8th':3,'9th':4,'10th':5,'11th':6,'12th':7,
                                     'HS-grad':8,'Some-college':9,'Assoc-voc':10,'Assoc-acdm':11,'Bachelors':12,
                                     'Masters':13,'Prof-school':14,'Doctorate':15})
    
input_data['age']=np.log1p(input_data['age'])
    
num_cols_to_standardize = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
input_data[num_cols_to_standardize] = scaler.fit_transform(input_data[num_cols_to_standardize])

#make the prediction
if st.button("Predict Income"):
    prediction = model.predict(input_data)
    result ="> 50k "if prediction[0]==1 else "< 50k"

    st.success(f"Predicted Income is: {result}")