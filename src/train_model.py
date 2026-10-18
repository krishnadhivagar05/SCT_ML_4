
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

dataset_path = "dataset/leapGestRecog"

data = []
labels = []
gesture_names = {}

for subject in sorted(os.listdir(dataset_path)):
    subject_path = os.path.join(dataset_path, subject)

    if not os.path.isdir(subject_path):
        continue

    for gesture in sorted(os.listdir(subject_path)):
        gesture_path = os.path.join(subject_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        if "_" not in gesture:
            continue

        gesture_name = gesture.split("_", 1)[1]

        if gesture_name not in gesture_names:
            gesture_names[gesture_name] = len(gesture_names)

        label = gesture_names[gesture_name]

        for filename in os.listdir(gesture_path):
            image_path = os.path.join(gesture_path, filename)

            try:
                image = Image.open(image_path).convert("L")
                image = image.resize((64, 64))
                image_array = np.array(image)

                features = hog(
                    image_array,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2)
                )

                data.append(features)
                labels.append(label)

            except Exception:
                continue

X = np.array(data)
y = np.array(labels)

print("Total images loaded:", len(X))
print("Number of gesture classes:", len(gesture_names))
print("Gesture classes:", gesture_names)

if len(X) == 0 or len(gesture_names) != 10:
    raise ValueError(
        "Expected 10 gesture classes. Check the dataset folder structure."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", class_weight="balanced")
)

print("Training the SVM model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Training completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

model_data = {
    "model": model,
    "gesture_names": gesture_names
}

joblib.dump(model_data, "model/hand_gesture_model.pkl")

labels_order = list(range(len(gesture_names)))
class_names = [
    next(name for name, number in gesture_names.items() if number == label)
    for label in labels_order
]

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels_order
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names
)

display.plot(xticks_rotation=45, cmap="Blues")
plt.title("Hand Gesture Recognition - SVM")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()

print("Model saved to model/hand_gesture_model.pkl")
print("Confusion matrix saved to results/confusion_matrix.png")
