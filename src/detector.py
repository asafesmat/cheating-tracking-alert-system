import os
import cv2
from datetime import datetime
from ultralytics import YOLO
from database import init_db, insert_alert

MODEL_PATH = "models/best.pt"
SCREENSHOT_DIR = "alerts/screenshots"
CONF = 0.25
SUSPICIOUS_SEC = 0.5
DECAY = 0.5                      # بينقص كام لما الكلاس يختفي (بدل التصفير)

CHEATING_SEC = {"Use_phone": 2.0, "Cheat_paper": 2.0, "Look_around": 3.0}
BEHAVIOR = {"Use_phone": "Phone detected",
            "Cheat_paper": "Cheat paper detected",
            "Look_around": "Looking at neighbor"}
GREEN, RED = (0, 255, 0), (0, 0, 255)


def _save_image(path, img):
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        buf.tofile(path)
    return ok


def process_video(video_path, stop_event):
    """Generator: بيطلّع (frame, events) لكل فريم."""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    init_db()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    model = YOLO(MODEL_PATH)     # موديل جديد لكل فيديو عشان الـ IDs ما تكمّلش
    names = model.names
    video_name = os.path.basename(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    susp_frames = SUSPICIOUS_SEC * fps
    cheat_frames = {c: s * fps for c, s in CHEATING_SEC.items()}

    score = {}        # (student ID, class) -> counter
    last_conf = {}    # (student ID, class) -> آخر ثقة
    alerted = set()   # (student ID, class) اللي اتسجّلت لها حادثة
    frame_idx = 0
    conf_sum, conf_n = {}, {}   # لحساب متوسط الثقة

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                            conf=CONF, verbose=False)[0]
            boxes = r.boxes

            current = {}   # student ID -> (box, class name, conf)
            if boxes.id is not None:
                for box, tid, cls, conf in zip(boxes.xyxy.int().tolist(),
                                               boxes.id.int().tolist(),
                                               boxes.cls.int().tolist(),
                                               boxes.conf.tolist()):
                    current[tid] = (box, names[cls], conf)

            # 1) أي عداد مش متطابق مع الفريم ده بينقص
            for key in list(score):
                tid, cname = key
                if not (tid in current and current[tid][1] == cname):
                    score[key] = max(0.0, score[key] - DECAY)

            # 2) العداد المتطابق بيزيد (للسلوكيات المشبوهة بس)
            for tid, (box, cname, conf) in current.items():
                if cname in CHEATING_SEC:
                    key = (tid, cname)
                    score[key] = score.get(key, 0.0) + 1
                    last_conf[key] = conf
                    conf_sum[key] = conf_sum.get(key, 0.0) + conf
                    conf_n[key] = conf_n.get(key, 0) + 1

            # 3) تحديد حالة كل طالب والرسم
            new_events = []
            for tid, (box, cname, conf) in current.items():
                status, color = "NORMAL", GREEN
                for (t, c), s in score.items():
                    if t != tid:
                        continue
                    if s >= cheat_frames[c]:
                        status, color = "CHEATING", RED
                        if (t, c) not in alerted:
                            alerted.add((t, c))
                            avg = conf_sum[(t, c)] / conf_n[(t, c)]
                            new_events.append((tid, c, avg))
                    elif s >= susp_frames and status != "CHEATING":
                        status = "SUSPICIOUS"

                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {tid} {cname} {status}", (x1, max(15, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # الحادثة بتتقفل لما العداد يرجع لصفر، والمتوسط بيبدأ من جديد
            for key in list(score):
                if score[key] == 0:
                    alerted.discard(key)
                    conf_sum.pop(key, None)
                    conf_n.pop(key, None)

            # 4) الصور والداتابيز (بعد الرسم عشان الصندوق الأحمر يبان)
            events = []
            for tid, cname, conf in new_events:
                now = datetime.now()
                path = f"{SCREENSHOT_DIR}/{now.strftime('%Y%m%d_%H%M%S')}_ID{tid}_{cname}.jpg"
                if _save_image(path, frame):
                    event = {
                        "time": now.isoformat(timespec="seconds"),
                        "student_id": tid,
                        "behavior": BEHAVIOR[cname],
                        "confidence": round(conf, 2),
                        "path": path,
                        "video_name": video_name,
                        "video_time": round(frame_idx / fps, 1),
                    }
                    insert_alert(event["time"], tid, event["behavior"],
                                 event["confidence"], path,
                                 video_name, event["video_time"])
                    events.append(event)

            yield frame, events
    finally:
        cap.release()