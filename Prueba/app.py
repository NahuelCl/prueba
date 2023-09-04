from flask import Flask, request, jsonify, render_template
import joblib  # Import joblib to load your pre-trained model
import numpy as np

app = Flask(__name__)

# Load your pre-trained classification model here
model = joblib.load('svc_trained.pkl')  # Replace with the path to your model file

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    Glucose = float(request.form['Glucose'])
    BloodPressure = float(request.form['BloodPressure'])
    Insulin = float(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = float(request.form['Age'])

    # Prepare the input data for classification
    input_data = np.array([[ Glucose, BloodPressure,Insulin, BMI, DiabetesPedigreeFunction,Age]])

    # Perform classification using the pre-trained model
    predicted_class = model.predict(input_data)[0]  # Get the prediction (0 or 1)

    # Define the result message based on the prediction
    if predicted_class == 1:
        result_message = "The person is classified as having diabetes."
    else:
        result_message = "The person is classified as not having diabetes."

    # Return the classification result as JSON
    return jsonify({'result': result_message})

if __name__ == '__main__':
    app.run(debug=True)

##Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age