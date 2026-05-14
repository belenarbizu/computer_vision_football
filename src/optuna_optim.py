import optuna
import supervision as sv
import numpy as np
import cv2
from ultralytics import YOLO
from reidentifier import PlayerReidentifier

VIDEO = "../videos/Untitled design.mp4"
EVALUATION_FRAMES = 300
model = YOLO("../models/yolov8_football.pt")

def objective(trial):
    conf                      = trial.suggest_float("conf", 0.15, 0.6)
    reid_threshold            = trial.suggest_float("reid_threshold", 0.55, 0.90)
    lost_track_buffer         = trial.suggest_int("lost_track_buffer", 10, 60)
    track_activation_threshold = trial.suggest_float("track_activation_threshold", 0.1, 0.5)
    minimum_matching_threshold = trial.suggest_float("minimum_matching_threshold", 0.6, 0.95)

    tracker = sv.ByteTrack(
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_matching_threshold=minimum_matching_threshold,
        frame_rate=30,
    )
    reid = PlayerReidentifier(device="cpu")

    known_ids     = set()
    remapped_ids  = {}
    unique_ids    = set()

    cap       = cv2.VideoCapture(VIDEO)
    frame_idx = 0

    while cap.isOpened() and frame_idx < EVALUATION_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        results    = model(frame, conf=conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, [1, 2])]
        detections = tracker.update_with_detections(detections)

        for i, tracker_id in enumerate(detections.tracker_id):
            tid  = int(tracker_id)
            bbox = detections.xyxy[i]
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]

            if tid not in known_ids and crop.size > 0:
                real_id = reid.search_in_gallery(crop, threshold=reid_threshold)
                if real_id is not None:
                    remapped_ids[tid] = real_id
                else:
                    reid.update_gallery(tid, crop)
                    known_ids.add(tid)

            final_id = remapped_ids.get(tid, tid)
            unique_ids.add(final_id)

        frame_idx += 1

    cap.release()

    n_ids = len(unique_ids)
    print(f"  Trial {trial.number:02d} | conf={conf:.2f} reid={reid_threshold:.2f} "
          f"buffer={lost_track_buffer} activ={track_activation_threshold:.2f} "
          f"match={minimum_matching_threshold:.2f} → {n_ids} IDs")
    return n_ids


optuna.logging.set_verbosity(optuna.logging.WARNING)  # silence internal logs

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("\n── Best parameters found ───────────────────")
print(f"  Minimum unique IDs: {study.best_value}")
for param, valor in study.best_params.items():
    if isinstance(valor, float):
        print(f"  {param}: {valor:.4f}")
    else:
        print(f"  {param}: {valor}")

# Parameter importance plot
try:
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
except Exception:
    print("\nInstall plotly to see the plot: pip install plotly")
