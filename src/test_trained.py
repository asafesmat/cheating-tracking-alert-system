import cv2
from ultralytics import YOLO

VIDEO_PATH = "Video/WhatsApp Video 2026-10-24 at 5.59.34 PM.mp4"       # غيّره لكل فيديو
model = YOLO("models/best.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO_PATH}")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                conf=0.05, verbose=False)[0]
    cv2.imshow("Trained model test", r.plot())
    if cv2.waitKey(10) & 0xFF == ord("q"):   
        break

cap.release()
cv2.destroyAllWindows()