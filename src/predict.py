
import numpy as np
import joblib

from PIL import Image
from skimage.feature import hog

image_path = input("Enter the image path: ")

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

    features = features.reshape(1, -1)

    saved_data = joblib.load("model/hand_gesture_model.pkl")

    model = saved_data["model"]
    gesture_names = saved_data["gesture_names"]

    prediction = model.predict(features)[0]

    gesture_name = next(
        name for name, label in gesture_names.items()
        if label == prediction
    )

    print("Predicted gesture:", gesture_name)

except FileNotFoundError:
    print("Image or model file not found. Check the path.")

except Exception as error:
    print("Error:", error)
