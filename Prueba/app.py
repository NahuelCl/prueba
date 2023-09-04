from flask import Flask, request, jsonify, render_template
import joblib  
import numpy as np

app = Flask(__name__)


model = joblib.load('svc_trained.pkl') 

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():

    Glucose = float(request.form['Glucose'])
    BloodPressure = float(request.form['BloodPressure'])
    Insulin = float(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = float(request.form['Age'])

   
    input_data = np.array([[ Glucose, BloodPressure,Insulin, BMI, DiabetesPedigreeFunction,Age]])

   
    predicted_class = model.predict(input_data)[0]  

   
    if predicted_class == 1:
        result_message = "La persona ha sido clasificada como que tiene diabetes."
    else:
        result_message = "La persona ha sido clasificada como que no tiene diabetes."

 
    return jsonify({'Clasificacion': result_message})

if __name__ == '__main__':
    app.run(debug=True)

##Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age