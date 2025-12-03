from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from db import get_connection
import subprocess
import sys

session_process = None

app = Flask(__name__)
app.secret_key = "supersecretkey"


@app.context_processor
def inject_now():
    return {"now": lambda: datetime.now()}


# =============================================
# HOME PAGE (Teacher Dashboard Only)
# =============================================
@app.route("/")
def index():
    if "username" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "teacher":
        return redirect(url_for("student_dashboard"))

    db = get_connection()
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT MAX(id) AS sid FROM sessions")
    row = cursor.fetchone()
    active_session_id = row["sid"] if row and row["sid"] else None

    total_students = 0
    total_gestures = 0
    top_gest = {}
    engagement_score = 0  # 🔥 Prevent undefined variable crash

    if active_session_id:
        # Total students
        cursor.execute("""
            SELECT COUNT(DISTINCT student_id) AS total
            FROM attendance
            WHERE session_id = %s
        """, (active_session_id,))
        total_students = cursor.fetchone()["total"]

        # Gesture counts
        cursor.execute("""
            SELECT gesture, COUNT(*) AS cnt
            FROM gestures
            WHERE session_id = %s
            GROUP BY gesture
        """, (active_session_id,))
        rows = cursor.fetchall()
        gesture_counts = {row["gesture"]: row["cnt"] for row in rows}
        total_gestures = sum(gesture_counts.values())

        top_gest = dict(sorted(gesture_counts.items(), key=lambda x: x[1], reverse=True))

        # Engagement Score
        positive_gestures = ["understood", "raise_hand", "wants_to_answer"]
        neutral_negative_gestures = ["repeat", "stop", "not_understood"]

        pos_count = sum(gesture_counts.get(g, 0) for g in positive_gestures)
        neg_count = sum(gesture_counts.get(g, 0) for g in neutral_negative_gestures)

        if total_gestures > 0:
            engagement_score = round((pos_count / total_gestures) * 100)

    status = "Running" if session_process else "Stopped"

    return render_template(
        "index.html",
        total_students=total_students,
        total_gestures=total_gestures,
        top_gest=top_gest,
        engagement_score=engagement_score,
        user=session["username"],
        role=session["role"],
        status=status,
    )


# =============================================
# STUDENT DASHBOARD
# =============================================
@app.route("/student_dashboard")
def student_dashboard():
    if "username" not in session or session.get("role") != "student":
        return redirect(url_for("login"))

    db = get_connection()
    cursor = db.cursor(dictionary=True)

    student = session["username"]

    cursor.execute("""
        SELECT * FROM attendance
        WHERE student_id = %s
        ORDER BY date DESC, time DESC
    """, (student,))
    attendance = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) AS cnt FROM attendance WHERE student_id = %s", (student,))
    total_attendance = cursor.fetchone()["cnt"]

    cursor.execute("SELECT COUNT(*) AS cnt FROM gestures WHERE student_id = %s", (student,))
    total_gestures = cursor.fetchone()["cnt"]

    cursor.execute("""
        SELECT gesture
        FROM gestures
        WHERE student_id = %s
        ORDER BY created_at DESC LIMIT 1
    """, (student,))
    last_gesture = cursor.fetchone()
    last_gesture = last_gesture["gesture"] if last_gesture else None

    return render_template(
        "student_dashboard.html",
        student=student,
        attendance=attendance,
        total_attendance=total_attendance,
        total_gestures=total_gestures,
        last_gesture=last_gesture
    )


# =============================================
# REGISTER
# =============================================
@app.route("/register", methods=["GET", "POST"])
def register():
    message = ""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        role = request.form.get("role")
        roll_no = request.form.get("roll_no") if role == "student" else None

        if not username or not password or not role:
            return render_template("register.html", message="All fields required!")

        db = get_connection()
        cursor = db.cursor(dictionary=True)

        if role == "teacher":
            cursor.execute("SELECT * FROM users WHERE username=%s AND role='teacher'", (username,))
            if cursor.fetchone():
                return render_template("register.html", message="Teacher already exists!")

        if role == "student":
            if not roll_no:
                return render_template("register.html", message="Roll number required!")
            cursor.execute("SELECT * FROM users WHERE roll_no=%s", (roll_no,))
            if cursor.fetchone():
                return render_template("register.html", message="Roll number already exists!")

        hashed_password = generate_password_hash(password)

        cursor.execute("""
            INSERT INTO users (username, password, role, roll_no)
            VALUES (%s, %s, %s, %s)
        """, (username, hashed_password, role, roll_no))
        db.commit()

        return redirect(url_for("login"))

    return render_template("register.html", message=message)


# =============================================
# LOGIN
# =============================================
@app.route("/login", methods=["GET", "POST"])
def login():
    message = ""
    if request.method == "POST":
        username = request.form.get("username")
        roll_no = request.form.get("roll_no")
        password = request.form.get("password")
        role = request.form.get("role")

        db = get_connection()
        cursor = db.cursor(dictionary=True)

        if role == "teacher":
            cursor.execute("SELECT * FROM users WHERE username=%s AND role='teacher'", (username,))
        else:
            cursor.execute("""
                SELECT * FROM users 
                WHERE username=%s AND roll_no=%s AND role='student'
            """, (username, roll_no))

        user = cursor.fetchone()

        if user and check_password_hash(user["password"], password):
            session["username"] = user["username"]
            session["role"] = user["role"]
            session["roll_no"] = user.get("roll_no")
            return redirect(url_for("index" if role == "teacher" else "student_dashboard"))

        message = "Invalid credentials!"

    return render_template("login.html", message=message)


# =============================================
# LOGOUT
# =============================================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# =============================================
# START SESSION
# =============================================
@app.route("/start_session")
def start_session():
    if "username" not in session or session.get("role") != "teacher":
        return redirect(url_for("login"))

    global session_process
    db = get_connection()
    cursor = db.cursor()

    cursor.execute("INSERT INTO sessions (teacher, start_time) VALUES (%s, NOW())",
                   (session["username"],))
    db.commit()

    new_session_id = cursor.lastrowid

    if session_process is None:
        session_process = subprocess.Popen([sys.executable, "realtime_full.py", str(new_session_id)])

    return redirect(url_for("index"))


# =============================================
# END SESSION
# =============================================
@app.route("/end_session")
def end_session():
    if "username" not in session or session.get("role") != "teacher":
        return redirect(url_for("login"))

    global session_process
    if session_process:

        db = get_connection()
        cursor = db.cursor()
        cursor.execute("UPDATE sessions SET ended_at = NOW() WHERE ended_at IS NULL")
        db.commit()

        session_process.terminate()
        session_process = None

    return redirect(url_for("index"))


# =============================================
# TEACHER PAGES (Attendance / Gestures / Live View)
# =============================================
@app.route("/attendance")
def attendance_page():
    if session.get("role") != "teacher":
        return redirect(url_for("student_dashboard"))

    db = get_connection()
    cursor = db.cursor(dictionary=True)

    # All attendance rows (for this teacher's sessions only)
    cursor.execute("""
        SELECT a.*
        FROM attendance a
        JOIN sessions s ON a.session_id = s.id
        WHERE s.teacher = %s
        ORDER BY a.date DESC, a.time DESC
    """, (session["username"],))
    attendance = cursor.fetchall()

    # Distinct students for filter dropdown
    cursor.execute("""
        SELECT DISTINCT a.student_id AS student_id
        FROM attendance a
        JOIN sessions s ON a.session_id = s.id
        WHERE s.teacher = %s
        ORDER BY a.student_id
    """, (session["username"],))
    students = [row["student_id"] for row in cursor.fetchall()]

    # This teacher's sessions for filter dropdown
    cursor.execute("""
        SELECT id, start_time
        FROM sessions
        WHERE teacher = %s
        ORDER BY start_time DESC
    """, (session["username"],))
    sessions = cursor.fetchall()

    return render_template(
        "attendance.html",
        attendance=attendance,
        students=students,
        sessions=sessions
    )

# =============================================
# Teacher: Gestures Page
# =============================================
@app.route("/gestures")
def gestures_page():
    if session.get("role") != "teacher":
        return redirect(url_for("student_dashboard"))

    db = get_connection()
    cursor = db.cursor(dictionary=True)

    # All gestures belonging to this teacher's sessions
    cursor.execute("""
        SELECT g.*
        FROM gestures g
        JOIN sessions s ON g.session_id = s.id
        WHERE s.teacher = %s
        ORDER BY g.created_at DESC
    """, (session["username"],))
    gestures = cursor.fetchall()

    # Distinct students for filter dropdown
    cursor.execute("""
        SELECT DISTINCT g.student_id AS student_id
        FROM gestures g
        JOIN sessions s ON g.session_id = s.id
        WHERE s.teacher = %s AND g.student_id IS NOT NULL
        ORDER BY g.student_id
    """, (session["username"],))
    students = [row["student_id"] for row in cursor.fetchall()]

    # This teacher's sessions for filter dropdown
    cursor.execute("""
        SELECT DISTINCT s.id, s.start_time
        FROM gestures g
        JOIN sessions s ON g.session_id = s.id
        WHERE s.teacher = %s
        ORDER BY s.start_time DESC
    """, (session["username"],))
    sessions = cursor.fetchall()

    # Gesture stats for charts (top gestures etc.)
    cursor.execute("""
        SELECT g.gesture, COUNT(*) AS cnt
        FROM gestures g
        JOIN sessions s ON g.session_id = s.id
        WHERE s.teacher = %s
        GROUP BY g.gesture
    """, (session["username"],))
    rows = cursor.fetchall()
    gesture_stats = {row["gesture"]: row["cnt"] for row in rows}

    return render_template(
        "gestures.html",
        gestures=gestures,
        students=students,
        sessions=sessions,
        gesture_stats=gesture_stats
    )

@app.route("/live_class")
def live_class():
    if session.get("role") != "teacher":
        return redirect(url_for("student_dashboard"))
    return render_template("live_class.html")


# =============================================
# API Routes (used by realtime_full.py)
# =============================================
@app.route("/api/mark_attendance", methods=["POST"])
def api_mark_attendance():
    student_id = request.form["student_id"]
    method = request.form["method"]
    session_id = request.form.get("session_id")

    db = get_connection()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO attendance (student_id, date, time, status, marked_by, session_id)
        VALUES (%s, CURDATE(), CURTIME(), 'Present', %s, %s)
    """, (student_id, method, session_id))
    db.commit()
    return "OK"


@app.route("/api/log_gesture", methods=["POST"])
def api_log_gesture():
    db = get_connection()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO gestures (student_id, gesture, confidence, session_id)
        VALUES (%s, %s, %s, %s)
    """, (
        request.form.get("student_id"),
        request.form["gesture"],
        request.form["confidence"],
        request.form.get("session_id")
    ))
    db.commit()
    return "OK"


@app.route("/api/live_data")
def live_data():
    db = get_connection()
    cursor = db.cursor(dictionary=True)

    cursor.execute("""
        SELECT 
            g.student_id AS student,
            g.gesture AS latest_gesture,
            DATE_FORMAT(g.created_at, '%H:%i:%s') AS last_seen
        FROM gestures g
        JOIN (
            SELECT student_id, MAX(created_at) AS latest
            FROM gestures
            GROUP BY student_id
        ) AS l ON g.student_id = l.student_id AND g.created_at = l.latest
    """)
    rows = cursor.fetchall()

    def engagement(g):
        if g in ["understood", "wants_to_answer", "raise_hand"]:
            return 90
        elif g in ["repeat", "stop"]:
            return 50
        return 10

    for row in rows:
        row["engagement"] = engagement(row["latest_gesture"] or "unknown")
        if row["student"] is None:
            row["student"] = "Unknown"

    return {"students": rows}


# =============================================
# RUN APP
# =============================================
if __name__ == "__main__":
    app.run(debug=True)
