import os
import sys
import numpy as np
from tensorflow import keras

def preprocess_image(image_path):
    """
    Preprocesses an image for MNIST digit classification.

    Steps:
    1. Load image in grayscale.
    2. Resize to 28x28 pixels.
    3. Convert to numpy array.
    4. Normalize pixel values to [0, 1].
    5. Add channel and batch dimensions.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        np.ndarray: Preprocessed image array with shape (1, 28, 28, 1).
    """
    # Load image with grayscale mode and target size 28x28
    img = keras.preprocessing.image.load_img(
        image_path, color_mode='grayscale', target_size=(28, 28)
    )

    # Convert image to numpy array
    img_array = keras.preprocessing.image.img_to_array(img)

    # Normalize pixel values to range [0, 1]
    img_array = img_array.astype('float32') / 255.0

    # Add batch dimension to make it (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

class DigitPredictor:
    """
    Handles loading a pre-trained MNIST model and predicting digits from images.
    """

    def __init__(self, model_path):
        """
        Initialize the predictor and load the pre-trained model.

        Args:
            model_path (str): Path to the .keras model file.
        """
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """Loads the Keras model from the specified path."""
        try:
            model = keras.models.load_model(self.model_path)
            print(f"Successfully loaded model from {self.model_path}")
            return model
        except Exception as e:
            print(f"Error loading model from {self.model_path}: {e}")
            raise

    def predict(self, processed_image):
        """
        Predicts the digit from a preprocessed image.

        Args:
            processed_image (np.ndarray): Preprocessed image array (1, 28, 28, 1).

        Returns:
            tuple: (predicted_digit, confidence_score)
        """
        # Make prediction
        prediction = self.model.predict(processed_image, verbose=0)

        # Get the index of the highest probability (the predicted digit)
        predicted_digit = np.argmax(prediction)

        # Get the probability of the predicted digit (confidence score)
        confidence_score = np.max(prediction) * 100

        return int(predicted_digit), float(confidence_score)

def main():
    """
    Main entry point for the Handwritten Digit Recognition System.
    """
    print("="*60)
    print("Handwritten Digit Recognition System")
    print("="*60)

    # Path to the pre-trained model
    model_path = os.path.join("mnist_model", "mnist_classifier.keras")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        # Initialize the predictor
        predictor = DigitPredictor(model_path)

        # Get image path from user
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            image_path = input("Please enter the path to the digit image (PNG/JPG): ").strip('"')

        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return

        # Step 1: Preprocess the image
        print(f"\nPreprocessing image: {image_path}...")
        processed_img = preprocess_image(image_path)

        # Step 2: Predict the digit
        print("Predicting digit...")
        digit, confidence = predictor.predict(processed_img)

        # Step 3: Display results
        print("\n" + "-"*30)
        print(f"RESULT")
        print(f"Predicted Digit: {digit}")
        print(f"Confidence:      {confidence:.2f}%")
        print("-"*30)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
