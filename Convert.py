import tensorflow as tf

# Load the Keras model
keras_model_path = r"C:\Users\hnaga\Downloads\irmodel.keras"  # Path to your existing .keras model
model = tf.keras.models.load_model(keras_model_path)

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the .tflite model
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Model converted and saved to {tflite_model_path}")
