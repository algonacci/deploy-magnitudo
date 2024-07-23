import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from flask_cors import cross_origin

app = Flask(__name__)

# Load model
model = load_model('model/earthquake_model.h5')

# Normalization parameters
x_min = np.array([0, -77, -180, 0])
x_max = np.array([1577664000, 86, 180, 700])
y_min = 0 
y_max = 10

def map_date_to_time(date_str):
    epoch = datetime(1970, 1, 1)
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (dt - epoch).total_seconds()

@app.route('/')
def home():
    return render_template('index.html')

@cross_origin
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    date = map_date_to_time(data['date'])
    latitude = float(data['latitude'])
    longitude = float(data['longitude'])
    depth = float(data['depth'])

    input_data = np.array([[date, latitude, longitude, depth]], dtype=np.float32)
    input_normalized = (input_data - x_min) / (x_max - x_min)

    # Prediction
    predicted_normalized = model.predict(input_normalized)
    print(predicted_normalized)
    predicted = predicted_normalized * (y_max - y_min) + y_min

    response = {
        'predictedMagnitude': float(predicted[0][0]),
        'predictionAccuracy': float(np.random.uniform(85, 95))  # Dummy accuracy
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
