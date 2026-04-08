import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from model import build_model

X = np.load("../data/features/X.npy")
y = np.load("../data/features/y.npy")

le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = build_model((X.shape[1],1), y.shape[1])

model.fit(X_train, y_train, epochs=10, batch_size=32)

model.save("../models/best_model.h5")
np.save("../models/labels.npy", le.classes_)
