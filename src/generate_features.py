import os
import numpy as np
from feature_extraction import build_dataset

# Get current file directory (src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go to project root → then data/raw
data_path = os.path.join(BASE_DIR, "..", "data", "raw")
feature_path = os.path.join(BASE_DIR, "..", "data", "features")

# Create features folder if not exists
os.makedirs(feature_path, exist_ok=True)

# Build dataset
X, y = build_dataset(data_path)

# Save features
np.save(os.path.join(feature_path, "X.npy"), X)
np.save(os.path.join(feature_path, "y.npy"), y)

print("✅ Features saved successfully!")