from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from reidentifier import PlayerReidentifier
import os
import json
import argparse

# ── Argument Parsing ──────────────────────────────────────
parser = argparse.ArgumentParser(description="Football Tactical Analysis Pipeline")
parser.add_argument("--video", type=str, default="../videos/Untitled design.mp4", help="Path to the video file.")
args = parser.parse_args()

VIDEO = args.video
if not os.path.exists(VIDEO):
    print(f"Error: The video file '{VIDEO}' does not exist.")
    exit()

model = YOLO("../models/yolov8_football.pt")
reid  = PlayerReidentifier(device="cpu")

# ── Load Homography Points ────────────────────────────────
try:
    with open("points.json", "r") as f:
        points_data = json.load(f)
    src_points = np.array(points_data["src_points"], dtype=np.float32)
    dst_points = np.array(points_data["dst_points"], dtype=np.float32)
except FileNotFoundError:
    print("Error: 'points.json' not found. Please run 'get_points.py' first.")
    exit()

H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

# ── Helpers ───────────────────────────────────────────────

def get_jersey_color(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = y2 - y1, x2 - x1
    torso = frame[
        y1 + int(h*0.25) : y1 + int(h*0.55),
        x1 + int(w*0.15) : x2 - int(w*0.15)
    ]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_not(
        cv2.inRange(hsv, np.array([35,40,40]), np.array([85,255,255]))
    )
    if cv2.countNonZero(mask) < 30:
        return None
    return np.array(cv2.mean(torso, mask=mask)[:3])

def crop_player(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None

def draw_field(ax):
    ax.set_facecolor("#2d7a2d")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_aspect("equal")
    ax.axis("off")
    b = dict(color="white", linewidth=1.5)
    ax.plot([0,105,105,0,0], [0,0,68,68,0], **b)
    ax.plot([52.5,52.5], [0,68], **b)
    ax.add_patch(plt.Circle((52.5,34), 9.15, fill=False, **b))
    ax.plot([0,16.5,16.5,0],    [13.84,13.84,54.16,54.16], **b)
    ax.plot([105,88.5,88.5,105],[13.84,13.84,54.16,54.16], **b)
    ax.plot([0,5.5,5.5,0],      [24.84,24.84,43.16,43.16], **b)
    ax.plot([105,99.5,99.5,105],[24.84,24.84,43.16,43.16], **b)

def generate_heatmap(title, positions, output_path, cmap="hot"):
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    heatmap, _, _ = np.histogram2d(xs, ys, bins=[105,68], range=[[0,105],[0,68]])
    heatmap = gaussian_filter(heatmap.T, sigma=2)
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    draw_field(ax)
    ax.imshow(heatmap, extent=[0,105,0,68], origin="lower",
              cmap=cmap, alpha=0.7, aspect="auto")
    ax.set_title(title, color="white", fontsize=14)
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

# ── First pass ────────────────────────────────────────

print("First pass — collecting data...")
tracker = sv.ByteTrack(
    lost_track_buffer=21,
    track_activation_threshold=0.3659,
    minimum_matching_threshold=0.8609
)
history           = defaultdict(list)
player_colors     = defaultdict(list)

# Re-id: IDs that ByteTrack considers new but we already know
known_ids      = set()   # IDs seen for at least 5 frames
remapped_ids   = {}      # {new_id: real_id}

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 25

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results    = model(frame, conf=0.3)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, [1, 2])]
    detections = tracker.update_with_detections(detections)

    for i, tracker_id in enumerate(detections.tracker_id):
        bbox  = detections.xyxy[i]
        crop  = crop_player(frame, bbox)
        tid   = int(tracker_id)

        # ── Re-identification ──────────────────────────
        if tid not in known_ids and crop is not None:
            real_id = reid.search_in_gallery(crop, threshold=0.5854)
            if real_id is not None:
                # It's a player we already knew with another ID
                remapped_ids[tid] = real_id
                print(f"  Re-id: #{tid} → #{real_id}")
            else:
                # New player — add to gallery
                reid.update_gallery(tid, crop)
                known_ids.add(tid)
        elif tid in known_ids and crop is not None:
            # Update gallery with new views
            reid.update_gallery(tid, crop)

        # Use final ID (might have been remapped)
        final_id = remapped_ids.get(tid, tid)

        # ── Field position ──────────────────────────
        foot_x = (bbox[0] + bbox[2]) / 2
        foot_y = bbox[3]
        pixel_point = np.array([[[foot_x, foot_y]]], dtype=np.float32)
        x_meters, y_meters = cv2.perspectiveTransform(pixel_point, H)[0][0]
        if 0 <= x_meters <= 105 and 0 <= y_meters <= 68:
            history[final_id].append((x_meters, y_meters))

        # ── Jersey color ─────────────────────────────
        color = get_jersey_color(frame, bbox)
        if color is not None:
            player_colors[final_id].append(color)

    cv2.imshow("First Pass", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Unique IDs after re-id: {len(history)}")
print(f"Remappings performed:   {len(remapped_ids)}")

# ── K-means: clasificar equipos ───────────────────────────

print("\nClassifying teams...")
mean_colors = {
    tid: np.array(colors).mean(axis=0)
    for tid, colors in player_colors.items()
    if len(colors) >= 10
}

ids = list(mean_colors.keys())
X   = np.array([mean_colors[tid] for tid in ids])

kmeans   = KMeans(n_clusters=2, random_state=0, n_init=10)
labels = kmeans.fit_predict(X)

teams             = {0: [], 1: []}
team_assignment   = {}
for tid, team in zip(ids, labels):
    teams[team].append(tid)
    team_assignment[int(tid)] = int(team)

print("\n── Team Classification ────────────────")
for team, players in teams.items():
    center = kmeans.cluster_centers_[team].astype(int)
    print(f"Team {team} (BGR: {center}): {[int(p) for p in players]}")

# ── Heatmaps ──────────────────────────────────────────────

print("\nGenerating heatmaps...")
os.makedirs("../outputs", exist_ok=True)

for tid, positions in history.items():
    if len(positions) < 20:
        continue
    generate_heatmap(f"Player #{tid}", positions,
                    f"../outputs/player_{tid:03d}.png")

team_cmaps = {0: "Reds", 1: "Blues"}
for team, players in teams.items():
    positions = [p for tid in players for p in history[tid]]
    if len(positions) < 20:
        continue
    generate_heatmap(f"Team {team}", positions,
                    f"../outputs/team_{team}.png",
                    cmap=team_cmaps[team])
    print(f"  team {team} → outputs/team_{team}.png")

all_positions = [p for positions in history.values() for p in positions]
generate_heatmap("Global", all_positions, "../outputs/global.png")
print("  global → outputs/global.png")

# ── Segunda pasada: vídeo anotado ────────────────────────

print("\nSecond pass — generating annotated video...")
TEAM_COLOR = {
    0:  (50,  50,  200),
    1:  (200, 200, 200),
    -1: (128, 128, 128),
}

tracker2 = sv.ByteTrack(
    lost_track_buffer=21,
    track_activation_threshold=0.3659,
    minimum_matching_threshold=0.8609
)
cap2     = cv2.VideoCapture(VIDEO)
width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(
    "../outputs/team_outputs.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (width, height)
)

while cap2.isOpened():
    ret, frame = cap2.read()
    if not ret:
        break

    results    = model(frame, conf=0.5113)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, [1, 2])]
    detections = tracker2.update_with_detections(detections)

    if detections.tracker_id is not None:
        for i, tid in enumerate(detections.tracker_id):
            tid_int  = int(tid)
            final_id = remapped_ids.get(tid_int, tid_int)
            team     = team_assignment.get(final_id, -1)
            color    = TEAM_COLOR[team]
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"#{final_id} T{team}",
                        (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

    writer.write(frame)
    cv2.imshow("Teams", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap2.release()
writer.release()
cv2.destroyAllWindows()

print("\nReady:")
print("  - outputs/                 → heatmaps by player and team")
print("  - outputs/team_outputs.mp4 → annotated video with teams and re-id")
