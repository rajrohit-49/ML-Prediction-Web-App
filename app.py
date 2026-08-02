from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

# Feature names
FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst', 'age',
]

@app.route('/hello',methods=['GET'])
def hello():
    return '<h1>Hello! Welcome to the Breast Cancer Predictor App 👋</h1>'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            input_data = [float(request.form[feature]) for feature in FEATURES]
            input_data_np = np.array(input_data).reshape(1, -1)
            prediction = model.predict(input_data_np)

            result = 'Benign' if prediction[0] == 1 else 'Malignant'
            return render_template('index.html', result=result, input_data=request.form)
        except Exception as e:
            return render_template('index.html', result='Error: ' + str(e))

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
