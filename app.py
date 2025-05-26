from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

# Route for the homepage (form)
@app.route('/')
def home():
    return render_template('index.html')

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from form
        features = [float(request.form[key]) for key in request.form]

        # Convert to array and scale
        final_input = np.array([features])
        final_input_scaled = scaler.transform(final_input)

        # Predict using scaled input
        prediction = model.predict(final_input_scaled)
        result = "Fraud" if prediction[0] == 1 else "Not Fraud"
        return f"<h2>Prediction: {result}</h2><br><a href='/'>Try Again</a>"
    except Exception as e:
        return f"<h2>Error: {str(e)}</h2>"

if __name__ == '__main__':
    app.run(debug=True)
