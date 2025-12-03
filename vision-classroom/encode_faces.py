import cv2
import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

DATA_DIR = "face_dataset"
MODEL_FILE = "face_model_knn.pkl"
LABELS_FILE = "face_labels.pkl"

def encode_faces():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    features = []
    labels = []
    label_map = {}
    current_label = 0

    print("[INFO] Scanning images...")

    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root)

                if label not in label_map:
                    label_map[label] = current_label
                    current_label += 1

                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect face
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) == 0:
                    continue

                (x, y, w, h) = faces[0]
                face_roi = gray[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (100, 100))

                features.append(face_roi.flatten())
                labels.append(label_map[label])

    features = np.array(features)
    labels = np.array(labels)

    print(f"[INFO] Total faces: {len(features)}")

    print("[INFO] Training KNN model...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(features, labels)

    print("[INFO] Saving model...")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(knn, f)

    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_map, f)

    print("[DONE] KNN Face Model trained successfully!")

if __name__ == "__main__":
    encode_faces()
