import cv2
import mediapipe as mp
import numpy as np
import pickle
import requests
import pyttsx3
import time
import threading

# ---------- Load Model ----------
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("gesture_labels.pkl", "rb") as f:
    GESTURES = pickle.load(f)

print("Model Loaded | Classes:", GESTURES)

# ---------- Flask Logging ----------
def send_gesture_to_db(gesture_name):
    try:
        requests.post("http://127.0.0.1:5000/api/log_gesture", data={
            "student_id": None,
            "gesture": gesture_name,
            "confidence": 1.0
        })
        print(f"📌 Logged: {gesture_name}")
    except:
        print("❌ DB Error")

# ---------- Voice Engine ----------
engine = pyttsx3.init("sapi5")
engine.setProperty("rate", 165)
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)

def speak_async(text):
    def run():
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=run, daemon=True).start()

SPEAK_MAP = {
    "understood": "Student understood ✔",
    "not_understood": "Student did not understand ❌",
    "raise_hand": "Student has a doubt ✋",
    "wants_to_answer": "Student wants to answer ✌",
    "stop": "Please wait ✊",
    "repeat": "Please repeat the concept ☝"
}

# ---------- Mediapipe Setup ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ---------- Runtime Variables ----------
cap = cv2.VideoCapture(0)
gesture_history = []
last_spoken = ""
last_speech_time = 0

print("\n🎬 Gesture Recognition Running — Press Q to Quit\n")

# ---------- Main Loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_final = None

    if result.multi_hand_landmarks:
        for lm_set in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm_set, mp_hands.HAND_CONNECTIONS)

            data = []
            for lm in lm_set.landmark:
                data.extend([lm.x, lm.y, lm.z])

            pred_idx = model.predict(np.array(data).reshape(1, -1))[0]
            gesture_raw = GESTURES[pred_idx]
            gesture_final = gesture_raw

            gesture_display = gesture_raw.replace("_", " ").title()
            cv2.putText(frame, gesture_display, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

    # -------- Confidence Stability - Voting Filter -------- #
    gesture_history.append(gesture_final)
    if len(gesture_history) > 10:
        gesture_history.pop(0)

    stable_gesture = max(set(gesture_history), key=gesture_history.count)

    now = time.time()
    if stable_gesture and stable_gesture != last_spoken and now - last_speech_time > 1.8:
        print("Detected:", stable_gesture)

        speak_text = SPEAK_MAP.get(stable_gesture, stable_gesture)
        speak_async(speak_text)
        send_gesture_to_db(speak_text)

        last_spoken = stable_gesture
        last_speech_time = now

    cv2.imshow("🎓 Vision Classroom - Gesture Recognition 👋", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Program Closed ✨")
