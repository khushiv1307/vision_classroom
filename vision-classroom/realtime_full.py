import cv2
import numpy as np
import pickle
import mediapipe as mp
import requests
import pyttsx3
import time
import threading
from datetime import datetime
import csv
import os
from queue import Queue
import sys

# ============================================================
#               SESSION ID FROM FLASK (CLI ARG)
# ============================================================
# When started from Flask: python realtime_full.py <session_id>
SESSION_ID = sys.argv[1] if len(sys.argv) > 1 else None
print("🔗 Attached to session_id:", SESSION_ID)

# ============================================================
#               TUNABLE PERFORMANCE PARAMETERS
# ============================================================

CONFIRM_FRAMES = 15          # frames needed to confirm same student (for attendance)
FACE_DIST_THRESHOLD = 3000.0 # KNN distance threshold for "Unknown"
FRAME_SKIP = 1               # process every Nth frame (1 = process all)

# ============================================================
#               GLOBAL STATE
# ============================================================

seen_students = set()        # students whose attendance already sent to DB this session
frame_streak = {}            # name -> consecutive frames seen

# ================== FILES ==================
GESTURE_MODEL = "gesture_model.pkl"
GESTURE_LABELS = "gesture_labels.pkl"
FACE_MODEL = "face_model_knn.pkl"
FACE_LABELS_FILE = "face_labels.pkl"
ATTENDANCE_FILE = "attendance.csv"

# ================== LOAD MODELS ==================
gesture_model = pickle.load(open(GESTURE_MODEL, "rb"))
GESTURES = pickle.load(open(GESTURE_LABELS, "rb"))

face_model = pickle.load(open(FACE_MODEL, "rb"))
FACE_LABELS = pickle.load(open(FACE_LABELS_FILE, "rb"))
inv_face_labels = {v: k for k, v in FACE_LABELS.items()}

print("✔ Models Loaded Successfully")

# ============================================================
#                   TEXT-TO-SPEECH (TTS) SETUP
# ============================================================

engine = pyttsx3.init("sapi5")
engine.setProperty("rate", 160)

tts_queue: Queue[str] = Queue()

def tts_worker():
    """Dedicated worker so TTS never blocks the main webcam loop."""
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)
        tts_queue.task_done()

# Start TTS worker thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak_async(text: str):
    """Enqueue text to be spoken without blocking video."""
    if text and text.strip():
        tts_queue.put(text)

# Messages for classroom use
SPEAK_MAP = {
    "understood": "I understood the topic",
    "not_understood": "I did not understand",
    "raise_hand": "I have a doubt",
    "wants_to_answer": "I want to answer",
    "stop": "Please slow down",
    "repeat": "Please repeat the topic"
}

DISPLAY_MAP = {
    "understood": "Understood 👍",
    "not_understood": "Not Understood 👎",
    "raise_hand": "Raise Hand ✋",
    "wants_to_answer": "Answer ✌",
    "stop": "Stop ✊",
    "repeat": "Repeat ☝"
}

# ============================================================
#               LOCAL CSV BACKUP FOR ATTENDANCE
# ============================================================

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Student", "FirstSeen", "LastSeen", "Count"])

def mark_attendance_csv(student: str):
    """Local CSV backup for attendance."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ATTENDANCE_FILE, "r") as f:
        rows = list(csv.reader(f))

    updated = False
    for row in rows:
        if row and row[0] == student:
            row[2] = now
            row[3] = str(int(row[3]) + 1)
            updated = True
            break

    if not updated:
        rows.append([student, now, now, "1"])

    with open(ATTENDANCE_FILE, "w", newline="") as f:
        csv.writer(f).writerows(rows)

# ============================================================
#       API DB Logging: Attendance & Gestures to Flask
# ============================================================

def mark_attendance_db(student: str):
    """Send attendance with session_id to Flask API."""
    if not SESSION_ID:
        print("⚠ No valid session_id! Attendance NOT logged.")
        return
    if student == "Unknown":
        return

    try:
        requests.post(
            "http://127.0.0.1:5000/api/mark_attendance",
            data={
                "student_id": student,  # Student NAME (matches users.username)
                "method": "AI",
                "session_id": SESSION_ID
            }
        )
        print(f"📝 Attendance → {student} | session {SESSION_ID}")
    except Exception as e:
        print("⚠ Attendance DB Error:", e)


def log_gesture_db(student: str, gesture_raw: str):
    """Send gesture + session_id to Flask API."""
    if not SESSION_ID:
        return
    if student == "Unknown":
        return  # Skip logging gestures if no identity

    try:
        requests.post(
            "http://127.0.0.1:5000/api/log_gesture",
            data={
                "student_id": student,  # ALWAYS store student
                "gesture": gesture_raw,
                "confidence": 1.0,
                "session_id": SESSION_ID
            }
        )
        print(f"🎯 Gesture → {student}: {gesture_raw} | session {SESSION_ID}")
    except Exception as e:
        print("⚠ Gesture DB Error:", e)


# ============================================================
#               MEDIAPIPE HANDS & FACE DETECTOR
# ============================================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,     # TRACKING mode
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ============================================================
#               WARM-UP: AVOID FIRST-GESTURE FREEZE
# ============================================================

def warmup_everything():
    print("⏳ Warming up TTS, Mediapipe and models...")

    # 1) Warm up TTS once
    try:
        engine.say("Initializing vision classroom")
        engine.runAndWait()
    except Exception as e:
        print("TTS warmup error:", e)

    # 2) Warm up Mediapipe Hands on a dummy frame
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_rgb = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
        for _ in range(3):
            _ = hands.process(dummy_rgb)
    except Exception as e:
        print("Hands warmup error:", e)

    # 3) Warm up gesture model
    try:
        feat_dim = getattr(gesture_model, "n_features_in_", 63)
        dummy_g = np.zeros((1, feat_dim), dtype=np.float32)
        _ = gesture_model.predict(dummy_g)
    except Exception as e:
        print("Gesture model warmup error:", e)

    # 4) Warm up face model (KNN)
    try:
        feat_dim_f = getattr(face_model, "n_features_in_", 10000)
        dummy_f = np.zeros((1, feat_dim_f), dtype=np.float32)
        _ = face_model.kneighbors(dummy_f, n_neighbors=1)
    except Exception as e:
        print("Face model warmup error:", e)

    print("✅ Warmup complete.\n")

warmup_everything()

# ============================================================
#                   MAIN CAMERA LOOP
# ============================================================

cap = cv2.VideoCapture(0)
# Optional: force a smaller resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_spoken_gesture = None
last_speak_time = 0.0
current_student_for_gesture = "Unknown"
frame_counter = 0

print("🚀 Vision Classroom Running — Press Q to Quit\n")
seen_students.clear()
frame_streak.clear()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if FRAME_SKIP > 1 and (frame_counter % FRAME_SKIP != 0):
        # Just show video without heavy processing to keep feed smooth
        cv2.imshow("Vision Classroom System 🎓", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---------- FACE RECOGNITION (ALL FACES) ----------
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    current_student_for_gesture = "Unknown"
    first_known_found = False

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (100, 100)).flatten().reshape(1, -1)

        # KNN distance check
        try:
            distances, idxs = face_model.kneighbors(face_roi, n_neighbors=1)
            dist = float(distances[0][0])
        except Exception as e:
            print("KNN error:", e)
            dist = FACE_DIST_THRESHOLD + 1

        if dist > FACE_DIST_THRESHOLD:
            name = "Unknown"
        else:
            pred_idx = face_model.predict(face_roi)[0]
            name = inv_face_labels.get(pred_idx, "Unknown")

        # Draw every face box + name
        color = (0, 255, 255) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if name != "Unknown":
            # Local CSV backup
            mark_attendance_csv(name)

            # continuity count per student
            frame_streak[name] = frame_streak.get(name, 0) + 1

            # once per session per student + only after CONFIRM_FRAMES
            if frame_streak[name] >= CONFIRM_FRAMES and name not in seen_students:
                seen_students.add(name)
                mark_attendance_db(name)

        # For gesture association, pick the first known face
        if name != "Unknown" and not first_known_found:
            current_student_for_gesture = name
            first_known_found = True

    # ---------- GESTURE RECOGNITION (ONE HAND) ----------
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        coords = []
        for lm in result.multi_hand_landmarks[0].landmark:
            coords.extend([lm.x, lm.y, lm.z])

        try:
            gesture_idx = gesture_model.predict(
                np.array(coords).reshape(1, -1)
            )[0]
            gesture_raw = GESTURES[gesture_idx].strip().replace("?", "").replace("\x0e", "").replace("\x06", "").replace("\x04", "").replace("\x0f", "")
        except Exception as e:
            print("Gesture prediction error:", e)
            gesture_raw = None

        if gesture_raw:
            gesture_display = DISPLAY_MAP.get(gesture_raw, gesture_raw)

            cv2.putText(frame, gesture_display, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            now = time.time()
            if gesture_raw != last_spoken_gesture and (now - last_speak_time) > 2:
                speaker_name = (
                    current_student_for_gesture
                    if current_student_for_gesture != "Unknown"
                    else "Student"
                )
                speak_text = f"{speaker_name} says {SPEAK_MAP.get(gesture_raw, gesture_raw)}"
                speak_async(speak_text)

                # Log gesture in DB
                log_gesture_db(current_student_for_gesture, gesture_raw)
                last_spoken_gesture = gesture_raw
                last_speak_time = now
    else:
        last_spoken_gesture = None

    cv2.imshow("Vision Classroom System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================
#                    CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
# stop TTS worker
tts_queue.put(None)
print("\n✨ System Closed Successfully")


