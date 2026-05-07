from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

model = YOLO("../models/yolov8_football.pt")
tracker = sv.ByteTrack()

src_points = np.array([[636, 215], [1310, 271], [365, 284], [589, 307]], dtype=np.float32)
dst_points = np.array([
    [0,    13.8],
    [16.5, 13.8],
    [0,   24.84],
    [10.5,24.84],
], dtype=np.float32)

H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

video_capture = cv2.VideoCapture("../videos/Untitled design.mp4")
frames_per_second = video_capture.get(cv2.CAP_PROP_FPS) or 25
player_history = defaultdict(list)

# ── Main Loop ───────────────────────────────────────
while video_capture.isOpened():
    success, current_frame = video_capture.read()
    if not success:
        break

    detection_results = model(current_frame, conf=0.3)[0]
    detections_obj = sv.Detections.from_ultralytics(detection_results)
    detections_obj = detections_obj[np.isin(detections_obj.class_id, [1, 2])]
    detections_obj = tracker.update_with_detections(detections_obj)

    for i, track_id in enumerate(detections_obj.tracker_id):
        bounding_box = detections_obj.xyxy[i]
        foot_x = (bounding_box[0] + bounding_box[2]) / 2
        foot_y = bounding_box[3]

        pixel_point = np.array([[[foot_x, foot_y]]], dtype=np.float32)
        x_meters, y_meters = cv2.perspectiveTransform(pixel_point, H)[0][0]

        if 0 <= x_meters <= 105 and 0 <= y_meters <= 68:
            player_history[track_id].append((x_meters, y_meters))

    cv2.imshow("tracking", current_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

# ── Heatmap Functions ──────────────────────────────────
def draw_field(ax):
    ax.set_facecolor("#2d7a2d")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_aspect("equal")
    ax.axis("off")
    white_color = dict(color="white", linewidth=1.5)
    ax.plot([0,105,105,0,0], [0,0,68,68,0], **white_color)
    ax.plot([52.5,52.5], [0,68], **white_color)
    ax.add_patch(plt.Circle((52.5, 34), 9.15, fill=False, **white_color))
    ax.plot([0,16.5,16.5,0],   [13.84,13.84,54.16,54.16], **white_color)
    ax.plot([105,88.5,88.5,105],[13.84,13.84,54.16,54.16], **white_color)
    ax.plot([0,5.5,5.5,0],     [24.84,24.84,43.16,43.16], **white_color)
    ax.plot([105,99.5,99.5,105],[24.84,24.84,43.16,43.16], **white_color)

def generate_heatmap(track_id, positions, output_path):
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    heatmap, _, _ = np.histogram2d(xs, ys, bins=[105, 68], range=[[0,105],[0,68]])
    heatmap = gaussian_filter(heatmap.T, sigma=2)
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    draw_field(ax)
    ax.imshow(heatmap, extent=[0,105,0,68], origin="lower",
              cmap="hot", alpha=0.6, aspect="auto")
    ax.set_title(f"Player #{track_id}", color="white", fontsize=14)
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def generate_global_heatmap(player_history, output_path):
    """Combined heatmap of all players."""
    all_positions = [p for positions in player_history.values() for p in positions]
    generate_heatmap("all", all_positions, output_path)

# ── Generate Heatmaps ──────────────────────────────────────
os.makedirs("heatmaps", exist_ok=True)

print("\nGenerating heatmaps...")
for tid, positions in player_history.items():
    if len(positions) < 20:
        continue
    generate_heatmap(tid, positions, f"heatmaps/player_{tid:03d}.png")
    print(f"  player #{tid} → heatmaps/player_{tid:03d}.png")

generate_global_heatmap(player_history, "heatmaps/global.png")
print("  global → heatmaps/global.png")
print("Done")
