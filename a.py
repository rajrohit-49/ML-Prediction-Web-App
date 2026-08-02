from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form and convert to float
        features = [float(x) for x in request.form.values()]
        input_data = np.asarray(features).reshape(1, -1)
        prediction = model.predict(input_data)

        result = 'Benign' if prediction[0] == 1 else 'Malignant'
        return render_template('index.html', prediction_text=f'The Breast Cancer is {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)