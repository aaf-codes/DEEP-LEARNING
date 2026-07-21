import numpy as np
import tensorflow as tf

# 1. Create a small dataset (random input values using NumPy).
# We'll create a batch of 3 samples, each with 4 features.
np.random.seed(42)
input_data = np.random.rand(3, 4).astype(np.float32) * 10

print("--- Original Input Values ---")
print(input_data)
print("\n")

# 2. Apply Batch Normalization using TensorFlow.
# We set training=True so it computes the mean and variance of the current batch.
batch_norm_layer = tf.keras.layers.BatchNormalization()
bn_output = batch_norm_layer(input_data, training=True)

print("--- Batch Normalization Output ---")
print(bn_output.numpy())
print("\n")

# 3. Apply Layer Normalization using TensorFlow.
# Normalizes across the features for each individual sample.
layer_norm_layer = tf.keras.layers.LayerNormalization()
ln_output = layer_norm_layer(input_data)

print("--- Layer Normalization Output ---")
print(ln_output.numpy())
print("\n")

# 5. Explanations
print("--- Comparison & Explanation ---")
print("What Batch Normalization does:")
print("- It normalizes the inputs across the batch dimension for each feature independently, helping to stabilize and speed up training.")
print("\nWhat Layer Normalization does:")
print("- It normalizes the inputs across the feature dimension for each individual example in the batch, useful for sequence models.")
print("\nDifference between them:")
print("- Batch Norm relies on the batch statistics (mean/variance across samples), so it can perform poorly on very small batch sizes. Layer Norm computes statistics per sample, making it completely independent of the batch size.")
