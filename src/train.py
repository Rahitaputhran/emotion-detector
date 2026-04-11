import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X = np.load(os.path.join(BASE_DIR, "data/features/X.npy"))
y = np.load(os.path.join(BASE_DIR, "data/features/y.npy"))

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# StandardScaler strictly brings disparate numerical spaces into a balanced identical scale, curing SVM input tracking logic.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Built a classic Support Vector Machine to avoid deep neural network and random forest architectures as requested.
# Added C=10 and class_weight='balanced' to rigidly protect minority classes (like Neutral) from being swallowed by Sad/Angry margin overlaps.
svm_model = SVC(kernel='rbf', probability=True, C=10.0, class_weight='balanced', random_state=42)
svm_model.fit(X_train, y_train)

print(f"SVM Training Accuracy: {svm_model.score(X_train, y_train):.2f}")
print(f"SVM Verification Check: {svm_model.score(X_test, y_test):.2f}")

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
with open(os.path.join(BASE_DIR, "models", "svm_model.pkl"), "wb") as f:
    pickle.dump(svm_model, f)
with open(os.path.join(BASE_DIR, "models", "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

np.save(os.path.join(BASE_DIR, "models", "labels.npy"), le.classes_)

