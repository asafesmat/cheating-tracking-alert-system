import os
import queue
import threading
import tkinter as tk
import winsound
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from database import get_alerts, init_db
from detector import process_video

VIEW_W, VIEW_H = 800, 450


class App:
    def __init__(self, root):
        self.root = root
        root.title("Cheating Tracking and Alert System")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        init_db()

        self.stop_event = threading.Event()
        self.queue = queue.Queue(maxsize=3)
        self.worker = None
        self.error = None
        self.photo = None

        top = tk.Frame(root)
        top.pack(pady=8)
        self.btn_upload = tk.Button(top, text="Upload Video", width=16,
                                    command=self.upload_video)
        self.btn_upload.pack(side="left", padx=4)
        self.btn_stop = tk.Button(top, text="Stop", width=10,
                                  command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)

        self.video_label = tk.Label(root, bg="black", width=VIEW_W // 8,
                                    height=VIEW_H // 16)
        self.video_label.pack(padx=8)

        self.status = tk.StringVar(value="Upload an exam video to start.")
        tk.Label(root, textvariable=self.status, anchor="w").pack(fill="x", padx=8, pady=4)

        # ---- Alert History ----
        tk.Label(root, text="Alert History (double-click a row to open its screenshot)",
                 anchor="w").pack(fill="x", padx=8)
        cols = ("time", "behavior", "confidence", "screenshot")
        self.tree = ttk.Treeview(root, columns=cols, show="headings", height=6)
        for col, title, width in [("time", "Date / Time", 150),
                                  ("behavior", "Behavior", 130),
                                  ("confidence", "Confidence", 80),
                                  ("screenshot", "Screenshot", 420)]:
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="x", padx=8, pady=(0, 8))
        self.tree.bind("<Double-1>", self.open_screenshot)

        for created_at, behavior, conf, path in reversed(get_alerts()):
            self.add_history(created_at, behavior, conf, path)

    # ---------- history ----------
    def add_history(self, created_at, behavior, conf, path):
        # index 0 = الأحدث فوق. بنحمّل القديم الأول عشان الترتيب يطلع صح.
        self.tree.insert("", 0, values=(created_at.replace("T", " "), behavior,
                                        conf, path))

    def open_screenshot(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        path = self.tree.item(sel[0])["values"][3]
        if os.path.exists(path):
            os.startfile(os.path.abspath(path))
        else:
            messagebox.showwarning("Not found", f"Screenshot file not found:\n{path}")

    # ---------- notification ----------
    def notify(self, e):
        t = datetime.fromisoformat(e["time"]).strftime("%I:%M %p")
        win = tk.Toplevel(self.root)
        win.overrideredirect(True)          # من غير شريط عنوان
        win.attributes("-topmost", True)
        win.configure(bg="#b71c1c")

        msg = (f"Cheating Alert Detected\n"
               f"Behavior: {e['behavior']}\n"
               f"Time: {t}\n"
               f"Screenshot saved successfully.")
        tk.Label(win, text=msg, justify="left", fg="white", bg="#b71c1c",
                 font=("Segoe UI", 10, "bold"), padx=16, pady=12).pack()

        win.update_idletasks()
        w, h = win.winfo_width(), win.winfo_height()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"+{sw - w - 20}+{sh - h - 60}")

        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        win.after(4000, win.destroy)

    # ---------- video ----------
    def upload_video(self):
        path = filedialog.askopenfilename(
            title="Select exam video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        self.stop()
        if self.worker is not None:
            self.worker.join()
        self.stop_event.clear()
        self.error = None
        while not self.queue.empty():
            self.queue.get_nowait()

        self.btn_upload.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status.set("Processing...")
        self.worker = threading.Thread(target=self._work, args=(path,), daemon=True)
        self.worker.start()
        self.poll()

    def _work(self, path):
        try:
            for item in process_video(path, self.stop_event):
                while not self.stop_event.is_set():
                    try:
                        self.queue.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            self.error = str(e)

    def poll(self):
        try:
            frame, events = self.queue.get_nowait()
            self.show_frame(frame)
            for e in events:
                self.status.set(f"Cheating Alert: {e['behavior']} | "
                                f"ID {e['student_id']} | {e['time']}")
                self.add_history(e["time"], e["behavior"], e["confidence"], e["path"])
                self.notify(e)
        except queue.Empty:
            pass

        if self.worker.is_alive() or not self.queue.empty():
            self.root.after(30, self.poll)
        else:
            self.finish()

    def show_frame(self, frame):
        h, w = frame.shape[:2]
        scale = min(VIEW_W / w, VIEW_H / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.config(image=self.photo, width=0, height=0)

    def stop(self):
        self.stop_event.set()

    def finish(self):
        self.btn_upload.config(state="normal")
        self.btn_stop.config(state="disabled")
        if self.error:
            messagebox.showerror("Error", self.error)
            self.status.set("Failed.")
        else:
            self.status.set("Finished.")

    def on_close(self):
        self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()