from flask import Flask, render_template ,request
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open("classification_model.pkl","rb"))
scaler=pickle.load(open("scaler.pkl",'rb'))

@app.route('/')
def home():
    return render_template('income.html',prediction='')
@app.route('/predict',methods=['POST'])

def index():
    input_data={
        'age': int(request.form['age']),
        'education':request.form['education'],
        'sex':request.form['sex'],
        'capital-gain':request.form['capital-gain'],
        'capital-loss':float(request.form['capital-loss']),
        'hours-per-week':int(request.form['hours-per-week']),
        'marital_Never-married': float(request.form['marital_Never-married']),
    }
    input_df = pd.DataFrame([input_data])
    input_df['sex']=input_df['sex'].map({"Male":0,"Female":1})
    input_df['education']=input_df['education'].map({'Preschool':0,'1st-4th':1,'5th-6th':2,'7th-8th':3,'9th':4,'10th':5,'11th':6,'12th':7,
                                     'HS-grad':8,'Some-college':9,'Assoc-voc':10,'Assoc-acdm':11,'Bachelors':12,
                                     'Masters':13,'Prof-school':14,'Doctorate':15})
    
    input_df['age']=np.log1p(input_df['age'])
    
    num_cols_to_standardize = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    input_df[num_cols_to_standardize] = scaler.fit_transform(input_df[num_cols_to_standardize])
    
    predicted_class=model.predict(input_df)
    if predicted_class==0:
        predict="Income is less than 50k"
    else:
        predict="Income is greater than 50k"
        
    return render_template('income.html',prediction=predict)
if __name__ == '__main__':
    app.run(debug=True)
   