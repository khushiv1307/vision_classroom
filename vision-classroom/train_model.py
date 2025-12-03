import numpy as np
import os
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

DATASET_PATH = "gesture_dataset"
GESTURES = os.listdir(DATASET_PATH)
print("Gestures:", GESTURES)

mp_hands = mp.solutions.hands.Hands(static_image_mode=True)

X, y = [], []

for label, gesture in enumerate(GESTURES):
    gesture_folder = os.path.join(DATASET_PATH, gesture)
    print(f"Processing gesture: {gesture}")

    for file in os.listdir(gesture_folder):
        img = cv2.imread(os.path.join(gesture_folder, file))
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(rgb)

        if not result.multi_hand_landmarks:
            continue

        lm = result.multi_hand_landmarks[0].landmark
        landmarks = []
        for l in lm:
            landmarks.extend([l.x, l.y, l.z])

        X.append(landmarks)
        y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# KNN Classifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("gesture_labels.pkl", "wb") as f:
    pickle.dump(GESTURES, f)

print("🎉 Training complete! Saved model as gesture_model.pkl")
