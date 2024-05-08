import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)
data = pd.read_csv('diabetes.csv')
model = joblib.load("predict_diabetes.pkl")



@app.route('/predict', methods=['POST'])
def predict():
    Pregnancies = request.form.get('Pregnancies')
    Glucose = request.form.get('Glucose')
    BloodPressure = request.form.get('BloodPressure')
    SkinThickness = request.form.get('SkinThickness')
    Insulin = request.form.get('Insulin')
    BMI = request.form.get('BMI')
    DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
    Age = request.form.get('Age')

    input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], columns=[
                              'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    prediction = model.predict_diabetes(input_data)
    print(prediction)
    
    if prediction == 0:
      return "The person is not diabetic."
    else:
      return "The person is diabetic."

if __name__ == "__main__":
    app.run(debug=True, port=5001)
