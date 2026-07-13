import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)

# ==========================================
# Exercise 1 - Build Your MLP Using TensorFlow
# ==========================================
print("\n--- Exercise 1: Basic MLP with Binary Crossentropy ---")
# Dataset
X = np.array([
    [2, 50],
    [3, 60],
    [4, 65],
    [5, 70],
    [6, 75],
    [7, 80],
    [8, 85],
    [9, 90]
], dtype=np.float32)

y = np.array([
    0, 0, 0, 0, 1, 1, 1, 1
], dtype=np.float32)

# Build the Neural Network
model = Sequential([
    Dense(4, activation="relu", input_shape=(2,)),
    Dense(1, activation="sigmoid")
])

# View the Model
model.summary()

# Compile the Model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the Model
print("Training Model 1...")
history = model.fit(X, y, epochs=100, verbose=0)
print("Training complete.")

# Make Predictions
print("\nPredictions for X:")
prediction = model.predict(X, verbose=0)
print(prediction)

# Predict a New Student
new_student = np.array([[7, 78]], dtype=np.float32)
print("\nPrediction for new student (7 hrs, 78%):")
prediction_new = model.predict(new_student, verbose=0)
print(prediction_new)


# ==========================================
# Exercise 2 – Model with Mean Squared Error
# ==========================================
print("\n--- Exercise 2: Model with Mean Squared Error ---")

model_mse = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

model_mse.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

print("Training Model 2 (MSE)...")
history_mse = model_mse.fit(X, y, epochs=100, verbose=0)
print("Training complete.")

# Evaluate
loss_mse, acc_mse = model_mse.evaluate(X, y, verbose=0)

print("\n===== MSE MODEL =====")
print("Loss :", loss_mse)
print("Accuracy :", acc_mse)
print("\nPredictions for X (MSE):")
print(model_mse.predict(X, verbose=0))


# ==========================================
# Exercise 4 – Build model with Gradient Descent
# ==========================================
print("\n--- Exercise 4: Model with SGD and Normalization ---")

# Normalize
X_norm = np.copy(X)
X_norm[:,0] = X_norm[:,0] / 9
X_norm[:,1] = X_norm[:,1] / 90

model_sgd = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

# Gradient Descent Optimizer
optimizer = SGD(learning_rate=0.01)

model_sgd.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['accuracy']
)

print("Training Model 4 (SGD)...")
history_sgd = model_sgd.fit(X_norm, y, epochs=100, verbose=0)
print("Training complete.")

plt.figure(figsize=(8,5))
plt.plot(history_sgd.history['loss'])
plt.title("Gradient Descent using MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()

# Save the plot as well so we have a copy
plt.savefig(r"C:\Users\aafre\Desktop\DEEP LEARNING [TASK=1]\loss_plot.png")
print("\nPlot saved to 'loss_plot.png'. Displaying plot (close the window to finish).")
plt.show()
