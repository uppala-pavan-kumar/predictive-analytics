from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import pandas as pd  # Import pandas
from joblib import load

app = Flask(__name__)
run_with_ngrok(app)

app.debug = True# Enable debug mode

model = load(r"/content/drive/My Drive/Sunbase/trained_model.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict/", methods=['POST'])
def predict():
    # Use request.form.get('input_name') to fetch form data
    age = int(request.form.get('Age'))
    gender = request.form.get('Gender')
    location = request.form.get('Location')
    subscription_length = int(request.form.get('Subscription_Length_Months'))
    monthly_bill = float(request.form.get('Monthly_Bill'))
    total_usage = float(request.form.get('Total_Usage_GB'))
    
    # Create a pandas DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Location': [location],
        'Subscription_Length_Months': [subscription_length],
        'Monthly_Bill': [monthly_bill],
        'Total_Usage_GB': [total_usage]
    })
    
    prediction = model.predict(input_data)
    
    return render_template('index.html', prediction_text="Prediction: {}".format("Churned" if prediction[0] == 1 else "Not Churned"))

if __name__ == "__main__":
    app.run()
