# Vehicle Number Plate Detection System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Tesseract OCR](https://img.shields.io/badge/Tesseract_OCR-4285F4?style=for-the-badge&logo=google&logoColor=white)

> A real-time computer vision pipeline for automated license plate recognition.

## 📌 Overview
This project implements a robust real-time vehicle number plate detection system. Designed for accuracy across various vehicle types and challenging lighting conditions, it leverages advanced computer vision techniques to isolate, process, and extract text from license plates.

## ✨ Key Features
- **Real-Time Pipeline**: Built a high-performance detection pipeline using OpenCV contour detection and adaptive thresholding.
- **High Reliability**: Validated to work reliably across 5+ distinct vehicle types and variable environmental conditions.
- **Accurate Text Extraction**: Integrated Tesseract OCR utilizing advanced ROI (Region of Interest) cropping and perspective correction to ensure highly accurate text extraction even under varied lighting scenarios.

## 🏗️ Architecture & Techniques
- Image Pre-processing (Grayscale conversion, Gaussian Blur, Canny Edge Detection)
- Contour analysis and bounding box estimation
- Perspective transformation for angled plates
- PyTesseract integration for final Optical Character Recognition

## 🚀 Getting Started
```bash
# Clone the repository
git clone https://github.com/TarunRepo/Vehicle-Number-Plate-Detection.git

# Install dependencies
pip install -r requirements.txt

# Run the detection pipeline
python src/main.py --input video.mp4
```
