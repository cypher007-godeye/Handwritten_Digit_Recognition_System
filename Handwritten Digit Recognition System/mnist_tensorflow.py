"""
MNIST Digit Classification using TensorFlow
============================================
This script demonstrates building and training a neural network to classify
handwritten digits from the MNIST dataset.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_and_preprocess_data():
    """
    Load MNIST dataset and preprocess it for training.

    Returns:
        tuple: (x_train, y_train, x_test, y_test) - Preprocessed data
    """
    # Load MNIST dataset
    # 60,000 training images and 10,000 test images
    # Each image is 28x28 pixels with grayscale values (0-255)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    print(f"Training data shape: {x_train.shape}")  # (60000, 28, 28)
    print(f"Test data shape: {x_test.shape}")       # (10000, 28, 28)

    # Normalize pixel values to range [0, 1]
    # This helps with gradient descent convergence
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape from (28, 28) to (28, 28, 1) for CNN input
    # The extra dimension represents the single grayscale channel
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print(f"Reshaped training data: {x_train.shape}")  # (60000, 28, 28, 1)
    print(f"Reshaped test data: {x_test.shape}")       # (10000, 28, 28, 1)

    return x_train, y_train, x_test, y_test


def build_cnn_model():
    """
    Build a Convolutional Neural Network (CNN) model for digit classification.

    Architecture:
    - Conv2D layers extract spatial features (edges, curves, shapes)
    - MaxPooling reduces spatial dimensions and prevents overfitting
    - Flatten converts 2D feature maps to 1D vectors
    - Dense layers perform final classification

    Returns:
        keras.Model: Compiled CNN model
    """
    model = models.Sequential([
        # First Convolutional Block
        # 32 filters, each learning different features
        # kernel_size=3 means 3x3 pixel receptive field
        # 'same' padding preserves spatial dimensions
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                      input_shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        # MaxPooling halves the spatial dimensions
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),  # Randomly drop 25% of neurons to prevent overfitting

        # Second Convolutional Block
        # 64 filters learn more complex, abstract features
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Flatten 2D feature maps to 1D for dense layer input
        layers.Flatten(),

        # Fully connected layer for classification
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Higher dropout for more regularization

        # Output layer: 10 neurons for digits 0-9
        # softmax outputs probability distribution over classes
        layers.Dense(10, activation='softmax')
    ])

    # Compile model with Adam optimizer and sparse categorical crossentropy loss
    # Adam adapts learning rate per parameter for faster convergence
    # sparse_categorical_crossentropy handles integer labels (0-9)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_simple_nn_model():
    """
    Build a simple Feedforward Neural Network (FNN) model.

    This is a simpler alternative to CNN that flattens images directly.
    Less accurate but faster to train.

    Returns:
        keras.Model: Compiled FNN model
    """
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),  # Flatten 28x28 to 784
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    """
    Train the model and return training history.

    Args:
        model: Keras model to train
        x_train, y_train: Training data
        x_test, y_test: Test/validation data
        epochs: Number of training iterations
        batch_size: Samples per gradient update

    Returns:
        History: Training history object
    """
    # Early stopping to prevent overfitting
    # Stops training if validation loss doesn't improve for 3 epochs
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Learning rate reduction on plateau
    # Reduces learning rate when progress stalls
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )

    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)

    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,  # Use 10% of training data for validation
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on Test Set...")
    print("="*60)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    return history


def visualize_results(model, x_test, y_test, history=None):
    """
    Visualize model predictions and training history.

    Args:
        model: Trained Keras model
        x_test, y_test: Test data
        history: Training history (optional)
    """
    # Get model predictions
    predictions = model.predict(x_test[:10], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    # Create figure for predictions
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.imshow(x_test[i].squeeze(), cmap='gray')
        actual = y_test[i]
        predicted = predicted_labels[i]
        color = 'green' if actual == predicted else 'red'
        ax.set_title(f'Pred: {predicted}\nActual: {actual}', color=color)
        ax.axis('off')

    plt.suptitle('Model Predictions (Green=Correct, Red=Wrong)', fontsize=14)
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print("\nSaved prediction visualization to 'predictions.png'")
    plt.close()

    # Plot training history if available
    if history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Train Acc')
        ax1.plot(history.history['val_accuracy'], label='Val Acc')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Loss plot
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("Saved training history visualization to 'training_history.png'")
        plt.close()


def save_model(model, save_dir='mnist_model'):
    """
    Save the trained model to disk.

    Args:
        model: Trained Keras model
        save_dir: Directory to save the model
    """
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, 'mnist_classifier.keras'))
    print(f"\nModel saved to '{save_dir}/mnist_classifier.keras'")


def load_and_predict(model_path, image_path=None):
    """
    Load a saved model and make predictions.

    Args:
        model_path: Path to saved model
        image_path: Path to image for prediction (optional)
    """
    model = keras.models.load_model(model_path)
    print(f"Loaded model from '{model_path}'")

    if image_path:
        # Load and preprocess custom image
        img = keras.preprocessing.image.load_img(
            image_path, color_mode='grayscale', target_size=(28, 28)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        prediction = model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        print(f"Predicted digit: {predicted_digit}")
        print(f"Confidence: {confidence:.2f}%")

    return model


def main():
    """
    Main function to run the complete MNIST classification workflow.
    """
    print("="*60)
    print("MNIST Digit Classification with TensorFlow")
    print("="*60)

    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing MNIST dataset...")
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # Step 2: Build the model
    print("\n[Step 2] Building CNN model...")
    model = build_cnn_model()
    model.summary()

    # Step 3: Train the model
    print("\n[Step 3] Training the model...")
    history = train_model(model, x_train, y_train, x_test, y_test, epochs=15)

    # Step 4: Visualize results
    print("\n[Step 4] Visualizing results...")
    visualize_results(model, x_test, y_test, history)

    # Step 5: Save the model
    print("\n[Step 5] Saving the model...")
    save_model(model)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

    return model, history


if __name__ == "__main__":
    model, history = main()
