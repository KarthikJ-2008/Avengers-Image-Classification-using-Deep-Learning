# Avengers Image Classification using CNN

A deep learning project that classifies images of Avengers characters — **Captain America, Hulk, Iron Man, Spider-Man, and Thor** — using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Overview

🎯 Trains a CNN from scratch to recognize five Marvel superheroes from images.
🔄 Covers the full pipeline: loading and augmenting image data, building and training a CNN, saving the trained model, and running predictions on new/unseen images.

### Avengers Characters

![Avengers Characters](test/Avengers.jpg)

| Character | Class |
|---|---|
| 🛡️ Captain America | 0 |
| 💚 Hulk | 1 |
| ❤️ Iron Man | 2 |
| 🕷️ Spider-Man | 3 |
| ⚡ Thor | 4 |

## 🛠️ Tech Stack

🐍 **Python 3**
🧩 **TensorFlow / Keras** — model building and training
🔢 **NumPy** — array/image data handling
📈 **Matplotlib** — visualization
☁️ **Google Colab** (with Google Drive mounted for dataset/model storage)

## 🗂️ Dataset

📁 Images organized into class-labeled subfolders for training (`ImageDataGenerator.flow_from_directory`).
🖼️ **60 training images** across the 5 classes.
📐 Images resized to **128 × 128** pixels and pixel values rescaled to the `[0, 1]` range.
🔀 **Data augmentation** applied during training:
  - 🔄 Rotation range: 20°
  - 🔍 Zoom range: 0.2
  - ↔️ Horizontal flip: enabled

## 🧠 Model Architecture

A sequential CNN with three convolutional blocks followed by dense classification layers:

```
Input (128, 128, 3)
Conv2D(32, 3x3, ReLU) → MaxPooling2D(2x2)
Conv2D(64, 3x3, ReLU) → MaxPooling2D(2x2)
Conv2D(128, 3x3, ReLU) → MaxPooling2D(2x2)
Flatten
Dense(128, ReLU)
Dropout(0.5)
Dense(5, Softmax)
```

🔢 **Total parameters:** ~3.3M
⚙️ **Optimizer:** Adam
📉 **Loss function:** Categorical Crossentropy
🎯 **Metric:** Accuracy

## 🏋️ Training

📦 **Batch size:** 4
🔁 **Epochs:** 20
📈 Training accuracy improved from ~17% (epoch 1) to **~78%** by the final epoch, with the model peaking around **82%** accuracy mid-training.
💾 The trained model is saved as `superhero_cnn.keras`.

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install tensorflow numpy matplotlib
```

### 2️⃣ Train the model

🏃 Run the training notebook and allow the CNN to train for 20 epochs.

### 3️⃣ Test an image

✏️ Change the image path in the testing code:

```python
img_path = "path/to/your/test/image.jpg"
```

🚀 Then run the prediction code.

## 📊 Results

✅ The model achieves reasonable classification accuracy given a fairly small training set (60 images across 5 classes), correctly identifying all characters on test images.
📈 Performance can be further improved with a larger, more diverse dataset, transfer learning (e.g., using a pretrained backbone like MobileNet or ResNet), and additional regularization/tuning.

## ⭐ Support

🙌 If you found this project useful or helpful for learning Python, OpenCV, or OCR, consider giving the repository a ⭐.