from tensorflow.keras.models import load_model
import numpy as np
import os
import pandas as pd

# ✅ Load model
model = load_model('mp_hand_gesture.keras')

# ✅ Updated gesture list
GESTURES = [
    'okay', 'peace', 'thumbs_up', 'thumbs_down',
    'call_me', 'stop', 'rock', 'live_long',
    'fist', 'smile'
]

# ✅ Load a sample for prediction (you can change the filename here)
sample_file = 'gesture_data/stop.csv'
sample_data = pd.read_csv(sample_file, header=None).values
example_input = np.array([sample_data[0]])  # Pick the first row/sample

# ✅ Predict
predicted_probs = model.predict(example_input)
predicted_class = np.argmax(predicted_probs)
gesture_name = GESTURES[predicted_class]

print(f"🤖 Predicted gesture: {gesture_name}")
