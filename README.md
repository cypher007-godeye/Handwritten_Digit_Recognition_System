# MNIST Digit Classification using TensorFlow

A complete implementation of handwritten digit classification using Convolutional Neural Networks (CNN) with TensorFlow/Keras.

## Using the Recognition System

The system is designed to take a custom image of a handwritten digit and predict its value using the pre-trained CNN model.

### Quick Start
Run the recognition system and provide the path to your image:
```bash
python MNIST.py path/to/your_digit.png
```

### How it Works
1. **Preprocessing (`preprocess.py`)**:
   - Loads image as grayscale.
   - Resizes to 28x28 pixels.
   - Normalizes pixel values from [0-255] to [0-1].
   - Adds batch and channel dimensions to match the model's expected input shape `(1, 28, 28, 1)`.
2. **Prediction (`predictor.py`)**:
   - Loads the `mnist_classifier.keras` model.
   - Performs a forward pass to get probabilities for digits 0-9.
   - Returns the digit with the highest probability and its confidence percentage.
3. **Main Interface (`main.py`)**:
   - Orchestrates the flow and provides a user-friendly CLI output.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Code Logic & Workflow](#code-logic--workflow)
5. [Architecture Details](#architecture-details)
6. [Training Process](#training-process)
7. [Output Files](#output-files)
8. [Usage Examples](#usage-examples)

---

## Overview

**MNIST** is the "Hello World" of computer vision - a dataset of 70,000 grayscale images of handwritten digits (0-9). This project demonstrates:

- Loading and preprocessing image data
- Building a CNN architecture from scratch
- Training with callbacks (early stopping, learning rate scheduling)
- Model evaluation and visualization
- Saving and loading trained models

### Dataset Specifications

| Split | Images | Shape | Format |
|-------|--------|-------|--------|
| Training | 60,000 | 28x28 pixels | Grayscale (0-255) |
| Test | 10,000 | 28x28 pixels | Grayscale (0-255) |

---

## Project Structure

```
personal/
├── app.py                 # Flask Web Server
├── MNIST.py               # All-in-one CLI recognition system
├── mnist_tensorflow.py    # Training implementation code
├── README.md              # This documentation file
├── templates/             # Web frontend templates
│   └── index.html
├── static/                # Static assets (CSS/JS)
│   └── js/
│       └── script.js
└── mnist_model/           # Saved model directory
    └── mnist_classifier.keras
```

## Running the Web Interface

1. **Install requirements**:
   ```bash
   pip install flask opencv-python tensorflow numpy
   ```

2. **Start the server**:
   ```bash
   python app.py
   ```

3. **Access the UI**:
   Open your browser and go to `http://127.0.0.1:5000`

---

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install tensorflow numpy matplotlib
```

### Verify Installation

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## Code Logic & Workflow

### Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      START                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: LOAD DATA                                              │
│  - Download MNIST from Keras datasets                           │
│  - 60,000 training + 10,000 test images                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: PREPROCESS DATA                                        │
│  - Normalize pixels: 0-255 → 0.0-1.0                            │
│  - Reshape: (28, 28) → (28, 28, 1) add channel dimension        │
│  - Why? CNNs need explicit channel dimension                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: BUILD CNN MODEL                                        │
│  - Conv2D layers extract features (edges → shapes → digits)     │
│  - MaxPooling reduces dimensions                                │
│  - Dropout prevents overfitting                                 │
│  - Dense layer classifies into 10 categories                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: COMPILE MODEL                                          │
│  - Optimizer: Adam (adaptive learning rate)                     │
│  - Loss: Sparse Categorical Crossentropy                        │
│  - Metric: Accuracy                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: TRAIN MODEL                                            │
│  - Batch size: 128 images per gradient update                   │
│  - Epochs: 15 (with early stopping)                             │
│  - Validation split: 10%                                        │
│  - Callbacks: EarlyStopping, ReduceLROnPlateau                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: EVALUATE                                               │
│  - Test on unseen 10,000 images                                 │
│  - Expected accuracy: 98-99%                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: VISUALIZE                                              │
│  - Plot predictions vs actual                                   │
│  - Plot training accuracy/loss curves                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 8: SAVE MODEL                                             │
│  - Save to mnist_model/mnist_classifier.keras                   │
│  - Ready for inference/deployment                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       END                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Details

### CNN Model Architecture

```
Input Layer (28x28x1)
        │
        ▼
┌───────────────────────────────────────┐
│  CONV BLOCK 1                         │
│  ├─ Conv2D(32 filters, 3x3, ReLU)    │
│  ├─ Conv2D(32 filters, 3x3, ReLU)    │
│  ├─ MaxPooling2D(2x2) → 14x14x32     │
│  └─ Dropout(0.25)                    │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  CONV BLOCK 2                         │
│  ├─ Conv2D(64 filters, 3x3, ReLU)    │
│  ├─ Conv2D(64 filters, 3x3, ReLU)    │
│  ├─ MaxPooling2D(2x2) → 7x7x64       │
│  └─ Dropout(0.25)                    │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  CLASSIFICATION HEAD                  │
│  ├─ Flatten() → 3136 units           │
│  ├─ Dense(128, ReLU)                 │
│  ├─ Dropout(0.5)                     │
│  └─ Dense(10, Softmax) → Output      │
└───────────────────────────────────────┘
```

### Layer-by-Layer Explanation

| Layer | Purpose | Output Shape |
|-------|---------|--------------|
| Input | Raw 28x28 grayscale image | (None, 28, 28, 1) |
| Conv2D x2 | Learn low-level features (edges, corners) | (None, 28, 28, 32) |
| MaxPooling | Spatial downsampling | (None, 14, 14, 32) |
| Dropout | Regularization (25% dropout) | (None, 14, 14, 32) |
| Conv2D x2 | Learn high-level features (loops, curves) | (None, 14, 14, 64) |
| MaxPooling | Further downsampling | (None, 7, 7, 64) |
| Dropout | Regularization (25% dropout) | (None, 7, 7, 64) |
| Flatten | Convert to 1D vector | (None, 3136) |
| Dense | Feature combination | (None, 128) |
| Dropout | Regularization (50% dropout) | (None, 128) |
| Output | Digit classification (0-9) | (None, 10) |

---

## Training Process

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam | Adaptive learning rate, works well out-of-box |
| Learning Rate | 0.001 | Standard starting point for Adam |
| Loss Function | Sparse Categorical Crossentropy | Multi-class classification with integer labels |
| Batch Size | 128 | Balance between memory and gradient stability |
| Epochs | 15 | Maximum epochs (early stopping may end earlier) |
| Validation Split | 10% | Monitor overfitting during training |

### Callbacks Explained

**1. EarlyStopping**
```python
EarlyStopping(
    monitor='val_loss',      # Watch validation loss
    patience=3,              # Wait 3 epochs before stopping
    restore_best_weights=True # Revert to best model
)
```
- Prevents overfitting by stopping when validation loss stops improving
- Automatically reverts to best model weights

**2. ReduceLROnPlateau**
```python
ReduceLROnPlateau(
    monitor='val_loss',      # Watch validation loss
    factor=0.5,              # Halve learning rate on plateau
    patience=2,              # Wait 2 epochs
    min_lr=1e-7              # Minimum learning rate
)
```
- Reduces learning rate when training stalls
- Helps model converge to better minimum

### Expected Training Progress

```
Epoch 1/15  - Accuracy: 94%  - Val Accuracy: 97%
Epoch 2/15  - Accuracy: 97%  - Val Accuracy: 98%
Epoch 3/15  - Accuracy: 98%  - Val Accuracy: 98%
...
Final Test Accuracy: 98-99%
```

---

## Output Files

After running the script, these files are generated:

| File | Description |
|------|-------------|
| `predictions.png` | Grid of 10 test images with predicted vs actual labels |
| `training_history.png` | Plots of accuracy and loss over epochs |
| `mnist_model/mnist_classifier.keras` | Saved model for future use |

### Predictions Visualization Example

```
┌─────┬─────┬─────┬─────┬─────┐
│  5  │  0  │  4  │  1  │  9  │  ← Predicted
│Actual│Actual│Actual│Actual│Actual│
│  5  │  0  │  4  │  1  │  9  │  ← Actual
└─────┴─────┴─────┴─────┴─────┘
  ✓     ✓     ✓     ✓     ✓     ← Correct (green)
```

---

## Usage Examples

### Run Training

```bash
python mnist_tensorflow.py
```

### Load Saved Model and Predict

```python
from tensorflow import keras
import numpy as np

# Load the saved model
model = keras.models.load_model('mnist_model/mnist_classifier.keras')

# Load and preprocess a custom image
from tensorflow.keras.preprocessing import image
img = image.load_img('my_digit.png',
                      color_mode='grayscale',
                      target_size=(28, 28))
img_array = image.img_to_array(img)
img_array = img_array.astype('float32') / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
digit = np.argmax(prediction)
confidence = np.max(prediction) * 100

print(f"Predicted: {digit} ({confidence:.2f}% confidence)")
```

### Use Only the Simple Feedforward Network

In `mnist_tensorflow.py`, replace:
```python
model = build_cnn_model()
```
with:
```python
model = build_simple_nn_model()
```

Expected accuracy: ~97% (slightly lower but faster training)

---

## Key Concepts Explained

### Why Normalize Pixels to [0, 1]?

```
Raw pixels:     0-255 range → Large values cause unstable gradients
Normalized:     0.0-1.0 range → Stable gradients, faster convergence
```

### Why Add a Channel Dimension?

```
Original shape:  (28, 28)      → Ambiguous for CNN
Reshaped:        (28, 28, 1)   → Explicitly grayscale (1 channel)
RGB images:      (28, 28, 3)   → 3 channels (R, G, B)
```

### How Does Convolution Work?

```
Input Image (28x28)
    │
    ▼
┌─────────────────┐
│ 32 Conv Filters │  ← Each filter scans for different patterns
│    3x3 kernel   │
└─────────────────┘
    │
    ▼
Feature Maps (28x28x32)  ← Each channel highlights different features
    │
    ├─→ Channel 0: Vertical edges
    ├─→ Channel 1: Horizontal edges
    ├─→ Channel 2: Diagonal edges
    └─→ ... (29 more feature channels)
```

### Why Use Dropout?

```
During Training:
- Randomly "drop" (zero out) 25-50% of neurons each forward pass
- Forces network to learn redundant representations
- Prevents co-adaptation of neurons

During Inference:
- All neurons active (scaled by dropout rate)
- Ensemble effect from training improves generalization
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch_size to 64 or 32 |
| Low accuracy (<95%) | Train for more epochs, check data preprocessing |
| Overfitting (train >> val accuracy) | Increase dropout, add more data augmentation |
| Slow training | Use GPU, reduce model complexity |

---

## License

This project is for educational purposes. MNIST dataset is in the public domain.
