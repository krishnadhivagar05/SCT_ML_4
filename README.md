# Hand Gesture Recognition using HOG and SVM

## Overview

This project recognizes hand gestures from images using a Support Vector Machine (SVM) classifier. It was developed as part of the SkillCraft Technology Machine Learning Internship, Task 04.

## Dataset

The project uses the [LeapGestRecog dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).

- Total images: 20,000
- Gesture classes: 10
- Subjects: 10
- Image size used: 64 × 64 pixels

## Technologies Used

- Python
- NumPy
- Pillow
- Scikit-image
- Scikit-learn
- Matplotlib
- Joblib

## Algorithm

1. Load images and identify their gesture labels.
2. Convert images to grayscale and resize them to 64 × 64.
3. Extract Histogram of Oriented Gradients (HOG) features.
4. Split the data by subject to evaluate performance on unseen people.
5. Train an SVM classifier.
6. Evaluate the model using accuracy and a confusion matrix.
7. Save the trained model for predictions.

## Model Evaluation

- Feature extraction: HOG
- Classifier: Support Vector Machine
- Kernel: Linear
- Evaluation: Subject-independent train-test split
- Test accuracy: Update after running the final training script.

## Gesture Classes

- Palm
- L
- Fist
- Fist moved
- Thumb
- Index
- OK
- Palm moved
- C
- Down

## Project Structure

```text
SCT_ML_4/
├── dataset/
├── model/
├── results/
│   └── confusion_matrix.png
├── src/
│   ├── train_model.py
│   └── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Train the Model

Run from the project root:

```bash
python src/train_model.py
```

The trained model is saved to `model/hand_gesture_model.pkl`.

The confusion matrix is saved to `results/confusion_matrix.png`.

## Predict a Gesture

Run:

```bash
python src/predict.py
```

Enter the path to a gesture image when prompted. The program prints the predicted gesture.

## Internship

SkillCraft Technology — Machine Learning Internship, Task 04.