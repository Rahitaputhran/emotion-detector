import numpy as np
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model("models/best_model.h5")
labels = np.load("models/labels.npy")

print("Model summary:")
print(model.summary())
print("\nLabels:", labels)
print("Number of emotions:", len(labels))
print("Model output shape:", model.output_shape)
