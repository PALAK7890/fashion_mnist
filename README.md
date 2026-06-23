# FashionAI: Fashion-MNIST Image Classification using CNN and Streamlit

## Overview

FashionAI is an end-to-end deep learning project that classifies fashion items using a Convolutional Neural Network (CNN) trained on the Fashion-MNIST dataset. The project includes a web application built with Streamlit that allows users to upload images or capture images using a webcam for real-time predictions.

The application performs image preprocessing, runs inference using a trained CNN model, and displays prediction confidence along with the top predicted classes.

---

## Features

* Convolutional Neural Network implemented using PyTorch
* Trained on the Fashion-MNIST dataset
* Image upload functionality
* Webcam image capture functionality
* Top-3 prediction display
* Confidence score visualization
* Modern web interface built using Streamlit
* End-to-end deployment pipeline

---

## Dataset

The project uses the Fashion-MNIST dataset containing 70,000 grayscale images of clothing items across 10 categories.

Classes:

| Label | Category    |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

Image properties:

* Image size: 28 × 28
* Grayscale images
* 10 classes

---

## Model Architecture

CNN Architecture:

Input Layer

→ Conv2D (32 filters)

→ ReLU

→ Batch Normalization

→ Max Pooling

→ Conv2D (64 filters)

→ ReLU

→ Batch Normalization

→ Max Pooling

→ Flatten

→ Fully Connected Layer (128)

→ Dropout

→ Fully Connected Layer (64)

→ Dropout

→ Output Layer (10 classes)

---

## Technologies Used

* Python
* PyTorch
* Streamlit
* NumPy
* Pandas
* OpenCV
* Scikit-learn
* Matplotlib

---

## Installation

Clone the repository:

```bash
git clone https://github.com/PALAK7890/fashion_mnist.git
```

Move into the project directory:

```bash
cd fashion_mnist
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Project Structure

```text
fashion_mnist/
│
├── app.py
├── model.py
├── fashion_cnn.pth
├── requirements.txt
├── FASHION_MNIST.ipynb
└── README.md
```

---

## Results

The CNN model was trained and evaluated on the Fashion-MNIST dataset and achieved strong classification performance on test data.

The web application provides:

* Real-time predictions
* Confidence scores
* Top predicted classes
* Interactive image-based inference

---

## Limitations

The model is trained on Fashion-MNIST images, which are simplified grayscale images with clean backgrounds. Performance may decrease when using real-world images from webcams or external sources because of variations in lighting, orientation, and image complexity.

---

## Future Improvements

* Train on real-world fashion datasets
* Add Grad-CAM visualization for explainability
* Improve webcam inference
* Add model comparison dashboard
* Deploy using cloud services
* Add user authentication and prediction history

---

## Author

Palak

This project was built as a practical implementation of deep learning, computer vision, and model deployment concepts.
