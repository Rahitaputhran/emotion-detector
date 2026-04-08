import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X = np.load(os.path.join(BASE_DIR, "data/features/X.npy"))
y = np.load(os.path.join(BASE_DIR, "data/features/y.npy"))

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a deeply robust multi-tree classifier bypassing deep neural network math breakdowns.
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train)

print(f"Random Forest Training Accuracy: {rf_model.score(X_train, y_train):.2f}")
print(f"Random Forest Verification Check: {rf_model.score(X_test, y_test):.2f}")

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
with open(os.path.join(BASE_DIR, "models", "rf_model.pkl"), "wb") as f:
    pickle.dump(rf_model, f)

np.save(os.path.join(BASE_DIR, "models", "labels.npy"), le.classes_)

