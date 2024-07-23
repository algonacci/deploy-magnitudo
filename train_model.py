import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
from datetime import datetime
import os

# Set random seed for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Print working directory
print(os.getcwd())

# Define paths
dataset_path = os.getcwd() + "//dataset"
model_path = os.getcwd() + "//model//earthquake_model"
plot_path = os.getcwd() + "//plot"

# Functions
def map_date_to_time(x):
    try:
        dt = datetime.strptime(x, "%m/%d/%Y")
    except ValueError:
        dt = datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ")
    return (dt - epoch).total_seconds()

# Constants
epoch = datetime(1970, 1, 1)
num_features = 4

# Extract data from CSV
data_frame = pd.read_csv('C:/xampp/htdocs/earthquake_magnitude/dataset/earthquakes.csv')

# Preprocessing
data_frame['Date'] = data_frame['Date'].apply(map_date_to_time)
inputs = data_frame[['Date', 'Latitude', 'Longitude', 'Depth']].to_numpy()
outputs = data_frame['Magnitude'].to_numpy().reshape(-1, 1)

# Normalization
x_min = np.amin(inputs, axis=0)
x_max = np.amax(inputs, axis=0)
y_min = np.amin(outputs)
y_max = np.amax(outputs)
inputs_normalized = (inputs - x_min) / (x_max - x_min)
outputs_normalized = (outputs - y_min) / (y_max - y_min)

# Split data
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs_normalized, outputs_normalized, test_size=0.2, random_state=42)

# Model creation function for KerasRegressor
def create_model(num_neurons=3, dropout_rate=0.2, l2_reg=0.01, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(num_features,)))
    model.add(Dense(num_neurons, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_neurons, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_neurons, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# Hyperparameter tuning
model = KerasRegressor(model=create_model, epochs=10, batch_size=32, verbose=0)
param_grid = {
    'model__num_neurons': [3, 5],
    'model__dropout_rate': [0.2, 0.3],
    'model__l2_reg': [0.01, 0.02],
    'model__learning_rate': [0.001, 0.01]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(train_inputs, train_outputs)

# Best hyperparameters
best_params = grid_result.best_params_
print("Best parameters found: ", best_params)

# K-Fold cross-validation with the best parameters
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []
for train_index, val_index in kfold.split(train_inputs):
    kfold_train_inputs, kfold_val_inputs = train_inputs[train_index], train_inputs[val_index]
    kfold_train_outputs, kfold_val_outputs = train_outputs[train_index], train_outputs[val_index]
    
    model = create_model(
        num_neurons=best_params['model__num_neurons'],
        dropout_rate=best_params['model__dropout_rate'],
        l2_reg=best_params['model__l2_reg'],
        learning_rate=best_params['model__learning_rate']
    )
    model.fit(kfold_train_inputs, kfold_train_outputs, epochs=10, batch_size=32, verbose=0)
    
    val_predictions = model.predict(kfold_val_inputs)
    val_mse = mean_squared_error(kfold_val_outputs, val_predictions)
    cv_results.append(val_mse)

print("Cross-validation MSE scores: ", cv_results)
print("Mean cross-validation MSE: ", np.mean(cv_results))

# Train final model with best parameters
final_model = create_model(
    num_neurons=best_params['model__num_neurons'],
    dropout_rate=best_params['model__dropout_rate'],
    l2_reg=best_params['model__l2_reg'],
    learning_rate=best_params['model__learning_rate']
)
final_model.fit(train_inputs, train_outputs, epochs=10, batch_size=32, verbose=0)

# Ensure the trainability of all layers is correctly set before saving
def set_trainability(model: tf.keras.models.Model, trainable: bool):
    for layer in model.layers:
        layer.trainable = trainable
        if hasattr(layer, 'layers'):
            set_trainability(layer, trainable)

# Set trainability of the model layers
set_trainability(final_model, trainable=True)  # Set to the correct trainability

# Save the final model without the optimizer state
final_model.save('C:/xampp/htdocs/earthquake_magnitude/model/earthquake_model.h5', include_optimizer=False)
print("Final model saved as 'earthquake_model.h5'")

# Testing
while True:
    try:
        lat = float(input("Enter Latitude between -77 to 86: "))
        if -77 <= lat <= 86:
            break
        else:
            print("Latitude should be between -77 to 86.")
    except ValueError:
        print("Invalid input. Latitude should be a number.")

while True:
    try:
        long = float(input("Enter Longitude between -180 to 180: "))
        if -180 <= long <= 180:
            break
        else:
            print("Longitude should be between -180 to 180.")
    except ValueError:
        print("Invalid input. Longitude should be a number.")

while True:
    try:
        depth = float(input("Enter Depth between 0 to 700: "))
        if 0 <= depth <= 700:
            break
        else:
            print("Depth should be between 0 to 700.")
    except ValueError:
        print("Invalid input. Depth should be a number.")

date = input("Enter the date (Month/Day/Year format): ")
test_input = np.array([[map_date_to_time(date), lat, long, depth]], dtype=np.float32)
test_input_normalized = (test_input - x_min) / (x_max - x_min)

# Load the saved model
saved_model = load_model('C:/xampp/htdocs/earthquake_magnitude/model/earthquake_model.h5')

# Predict using the loaded model
predicted_normalized = saved_model.predict(test_input_normalized)
predicted = predicted_normalized * (y_max - y_min) + y_min

print("Predicted magnitude:", predicted[0][0])

# Calculate accuracy
tolerance = 0.1  # Define your tolerance level here
accurate_predictions = np.abs(predicted - outputs.mean()) <= tolerance
accuracy_percentage = np.mean(accurate_predictions) * 100

print(f"Accuracy within {tolerance} of actual magnitude: {accuracy_percentage:.2f}%")
