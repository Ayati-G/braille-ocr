# Braille Recognition System

## Overview

This project is an AI-powered Braille Recognition System that converts Braille text from images into readable English text. The system combines object detection and image classification techniques to identify individual Braille characters and reconstruct complete words or sentences.

The goal of this project is to improve accessibility by providing a simple way to interpret Braille text using computer vision and deep learning.

---

## Features

* Upload an image containing Braille text
* Detect individual Braille cells using YOLOv8
* Classify each Braille character using MobileNetV3
* Reconstruct words and sentences from detected characters
* Web-based interface built with HTML, CSS, and JavaScript
* FastAPI backend for model inference

---

## System Architecture

```text
Input Image
     ↓
YOLOv8 OBB Detector
     ↓
Braille Cell Detection
     ↓
Character Cropping
     ↓
MobileNetV3 Classifier
     ↓
Character Recognition
     ↓
Sentence Reconstruction
     ↓
Output Text
```

---

## Technologies Used

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* FastAPI
* Uvicorn

### Machine Learning

* YOLOv8 OBB
* MobileNetV3
* PyTorch
* Torchvision

### Computer Vision

* OpenCV
* Pillow (PIL)

### Dataset and Annotation

* Roboflow

---

## Project Structure

```text
project/
│
├── app.py
├── class_mapping.json
├── mobilenet_braille.pth
├── best.pt
│
├── website/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
└── dataset/
    ├── train/
    ├── valid/
    └── test/
```

---

## Installation

### Clone the Repository

```bash
git clone <repository-url>
cd braille-recognition
```

### Create Virtual Environment

```bash
python -m venv braille_env
```

### Activate Environment

Windows:

```bash
braille_env\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Backend

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Server will be available at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Using the Website

1. Open the website in a browser.
2. Upload an image containing Braille text.
3. Click the Detect button.
4. The image is sent to the FastAPI backend.
5. YOLO detects Braille cells.
6. MobileNet classifies each detected character.
7. The recognized text is displayed on the screen.

---

## Future Improvements

* Support for multi-line Braille text
* Improved sentence reconstruction
* Real-time webcam recognition
* Mobile application support
* Grade 2 Braille recognition
* Cloud deployment

---

## Contributors

* Vansh Kapil – Backend Development, Frontend Development, Documentation
* Ayati Gupta – Dataset Preparation, Machine Learning, Testing

---

## License

This project is developed for educational and research purposes.
