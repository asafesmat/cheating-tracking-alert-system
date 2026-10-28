# Cheating Tracking and Alert System

A desktop application that tracks students in exam videos, flags suspicious behavior with a trained YOLO model, and records each incident with a screenshot, a database entry and an on-screen notification.

Built as a project for the Instant AI program. This is a learning project evaluated on a small dataset and four short test clips, not a production system. See [Limitations](#limitations).

## What it does

1. The user uploads an exam video in a Tkinter desktop app.
2. A YOLO11s model, fine-tuned on a public exam-behavior dataset, detects each student and classifies the behavior: `Normal`, `Use_phone`, `Cheat_paper` or `Look_around`.
3. ByteTrack gives every student a stable ID, so behavior can be measured over time instead of per frame.
4. A rule layer turns per-frame predictions into incidents. A behavior must persist before it is treated as cheating, which filters out single-frame mistakes.
5. When an incident is confirmed, the app turns the student's box red, saves a screenshot, stores the event in SQLite, adds it to the Alert History table and shows a pop-up notification.

Students without a confirmed incident keep a green box.

## Requirements coverage

| Requirement from the project brief | Status |
|---|---|
| Track students in exam video (YOLO + ByteTrack) | Done |
| Red box for cheating, green box for normal | Done |
| Screenshot saved locally on each incident | Done (`alerts/screenshots/`) |
| Database with date/time, behavior, confidence, screenshot path | Done, using SQLite (`alerts/alerts.db`) |
| Tkinter desktop app with an Upload Video button | Done |
| Alert history | Done |
| Immediate notification | Done (pop-up window and system sound) |
| Screenshot viewer | Done (double-click a row in the history table) |
| Firebase Firestore / Storage (recommended in the brief) | Not implemented; SQLite is used instead |
| Custom app icon (optional) | Not implemented |

## How incidents are decided

For every tracked student and every suspicious class, the app keeps a counter:

- It increases by 1 for each frame in which the student is predicted as that class.
- It decreases by 0.5 for each frame in which the prediction does not match, instead of resetting to zero. This keeps a real incident alive through the frame-to-frame flicker of the model.
- The student is marked suspicious after 0.5 seconds and a cheating alert is raised after:

| Behavior | Time needed for an alert |
|---|---|
| `Use_phone` | 2 s |
| `Cheat_paper` | 2 s |
| `Look_around` | 3 s |

`Look_around` gets the longest time because a short glance is not cheating. One alert is raised per incident, and the incident closes when the counter returns to zero.

## Model and dataset

- **Dataset:** [exam_cheating](https://universe.roboflow.com/behavior-cheating/exam_cheating-keaor) by behavior-cheating on Roboflow Universe, version 2, license CC BY 4.0. 772 images, four classes: `Cheat_paper`, `Look_around`, `Normal`, `Use_phone`.
- **Re-split:** the original validation and test splits had only 3-4 examples of the phone and paper classes, so the images were re-split (train 581 / valid 103 / test 88). Frames that came from video files were kept in the training set to reduce near-duplicate leakage between splits. The splitting code is in the notebook.
- **Model:** YOLO11s (Ultralytics), initialized from COCO-pretrained weights, 100 epochs, image size 640, batch size 16, seed 42, trained on a Colab T4 GPU in about 17 minutes.
- **Training notebook:** `notebooks/Cheating_Tracking.ipynb`.

### Validation results (103 images)

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| All | 207 | 0.877 | 0.847 | 0.885 | 0.549 |
| Cheat_paper | 17 | 0.992 | 1.000 | 0.995 | 0.524 |
| Look_around | 77 | 0.851 | 0.870 | 0.898 | 0.633 |
| Normal | 88 | 0.756 | 0.636 | 0.780 | 0.553 |
| Use_phone | 25 | 0.909 | 0.880 | 0.866 | 0.487 |

How to read these numbers:

- The validation set is small. `Cheat_paper` has only 17 instances, so its near-perfect score is not reliable evidence of strong performance.
- Some images in the dataset come from the same recording sessions, so the validation images are probably easier than genuinely new footage.
- The 88-image test split was not used for a final evaluation.

Training curves and the normalized confusion matrix are in `assets/`.

## Testing on real footage

The app was tried on four short exam clips (three of about 10 seconds, one of about 20 seconds). These clips are not included in the repository. The evaluation was qualitative: the clips were watched and the alerts compared with what happened on screen. No precision or recall figures were computed on them.

What was observed:

- **Phone use** was detected consistently, with a stable confidence of about 0.66-0.69.
- **Looking at a neighbor's paper** was detected as `Look_around` in one clip.
- Normal students were classified as `Normal` in most cases.

The timing thresholds were adjusted while watching these same clips, so they are tuned by hand and not independently validated.

## Limitations

- **False alerts on `Look_around`.** In one clip, a student leaning down to write was classified as `Look_around` and triggered an alert. The average confidence of that false alert (0.68) was higher than that of the true alert in another clip (0.51), so a confidence threshold could not separate them. Raising the threshold removed the true alert first. Fixing this needs more varied training examples, not parameter tuning.
- **Missed cases.** A student who appeared to copy from the student seated in front of him was not detected during the incident, even at a confidence of 0.05. This situation is probably not covered by the training data.
- **Small evaluation.** Four short clips and a 103-image validation set are not enough to estimate real-world accuracy.
- **Windows only.** The notification sound uses `winsound`.
- **Not a decision system.** The output is a prompt for a human to review a screenshot. It should not be used to accuse a student without human review.

## Project structure

```
.
├── src/
│   ├── app.py            # Tkinter app: upload, video view, alert history, notification
│   ├── detector.py       # detection, tracking, incident rules, screenshots
│   ├── database.py       # SQLite storage
│   └── test_trained.py   # view raw model predictions on a video
├── models/best.pt        # trained YOLO11s weights
├── notebooks/            # Colab training notebook
├── assets/               # training curves and confusion matrix
└── requirements.txt
```

`alerts/` (screenshots and `alerts.db`) is created automatically on the first run.

## Installation and usage

Developed and tested on Windows with Python 3.14.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/app.py
```

On the first run, Ultralytics may install the `lap` package needed by the tracker automatically. If it does, run the command again.

Click **Upload Video** and select an exam video. Incidents appear in the Alert History table. Double-click a row to open its screenshot.

## Credits

- Dataset: *exam_cheating* by behavior-cheating, Roboflow Universe, CC BY 4.0. https://universe.roboflow.com/behavior-cheating/exam_cheating-keaor
- Detection and tracking: [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO11 and ByteTrack.

Author: Asaf Esmat Wagih ([github.com/asafesmat](https://github.com/asafesmat)) ([linkedin.com/asafesmat](https://www.linkedin.com/in/asaf-esmat-557004357/))
