import cv2
import os

DATA_DIR = "face_dataset"

def capture_for_student(student_id, max_images=30):
    student_dir = os.path.join(DATA_DIR, student_id)
    os.makedirs(student_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    print(f"[INFO] Capturing images for {student_id}. Press SPACE to save, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"ID: {student_id} Count: {count}/{max_images}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Capture Faces", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            img_path = os.path.join(student_dir, f"{count+1}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"[SAVED] {img_path}")
            count += 1
            if count >= max_images:
                break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Face capture complete.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python capture_faces.py <student_id> [max_images]")
    else:
        sid = sys.argv[1]
        mi = int(sys.argv[2]) if len(sys.argv) >= 3 else 30
        capture_for_student(sid, mi)
