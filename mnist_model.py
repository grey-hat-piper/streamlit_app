import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# === Step 1: Load the MNIST dataset ===
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to 0–1
x_train = x_train / 255.0
x_test = x_test / 255.0

# === Step 2: Build the model ===
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),    # Flatten 28x28 images into a vector
    layers.Dense(128, activation='relu'),    # Dense hidden layer
    layers.Dropout(0.2),                     # Dropout for regularization
    layers.Dense(10, activation='softmax')   # Output layer for 10 digit classes
])

# === Step 3: Compile the model ===
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === Step 4: Train the model .===
print("📊 Training model...")
model.fit(x_train, y_train, epochs=5)

# === Step 5: Evaluate on test data ===
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

# === Step 6: Save the model ===
model.save("my_mnist_model.h5")
print("💾 Model saved as my_mnist_model.h5")
