from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
import os

VIDEO = "Untitled design.mp4"

model = YOLO("yolov8_football.pt")

src_points = np.array([[636, 215], [1310, 271], [365, 284], [589, 307]], dtype=np.float32)
dst_points = np.array([
    [0,    13.8 ],
    [16.5, 13.8 ],
    [0,    24.84],
    [10.5, 24.84],
], dtype=np.float32)
H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

# ── Functions ─────────────────────────────────────────────
def get_jersey_color(current_frame, bounding_box):
    x1, y1, x2, y2 = map(int, bounding_box)
    h, w = y2 - y1, x2 - x1
    torso_region = current_frame[
        y1 + int(h * 0.25) : y1 + int(h * 0.55),
        x1 + int(w * 0.15) : x2 - int(w * 0.15)
    ]
    if torso_region.size == 0:
        return None
    hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35,40,40]), np.array([85,255,255]))
    mask = cv2.bitwise_not(green_mask)
    if cv2.countNonZero(mask) < 30:
        return None
    return np.array(cv2.mean(torso_region, mask=mask)[:3])

def draw_field(ax):
    ax.set_facecolor("#2d7a2d")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_aspect("equal")
    ax.axis("off")
    white_line_style = dict(color="white", linewidth=1.5)
    ax.plot([0,105,105,0,0], [0,0,68,68,0], **white_line_style)
    ax.plot([52.5,52.5], [0,68], **white_line_style)
    ax.add_patch(plt.Circle((52.5, 34), 9.15, fill=False, **white_line_style))
    ax.plot([0,16.5,16.5,0],    [13.84,13.84,54.16,54.16], **white_line_style)
    ax.plot([105,88.5,88.5,105],[13.84,13.84,54.16,54.16], **white_line_style)
    ax.plot([0,5.5,5.5,0],      [24.84,24.84,43.16,43.16], **white_line_style)
    ax.plot([105,99.5,99.5,105],[24.84,24.84,43.16,43.16], **white_line_style)

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

# ── First Pass: Collect Positions and Colors ──────────

print("First pass — collecting data...")
tracker = sv.ByteTrack()
player_history = defaultdict(list)
player_colors = defaultdict(list)

video_capture = cv2.VideoCapture(VIDEO)
frames_per_second = video_capture.get(cv2.CAP_PROP_FPS) or 25

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
        detected_color = get_jersey_color(current_frame, bounding_box)
        if detected_color is not None:
            player_colors[track_id].append(detected_color)

    cv2.imshow("First Pass", current_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

# ── K-means: Classify Teams ───────────────────────────

print("Classifying teams...")
average_colors = {
    tid: np.array(colors).mean(axis=0)
    for tid, colors in player_colors.items()
    if len(colors) >= 10
}

player_ids = list(average_colors.keys())
X = np.array([average_colors[tid] for tid in player_ids])

kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
labels = kmeans.fit_predict(X)

teams = {0: [], 1: []}
team_assignment = {}
for tid, team in zip(player_ids, labels):
    teams[team].append(tid)
    team_assignment[int(tid)] = int(team)

print("\n── Team Classification ────────────────")
for team, players in teams.items():
    center = kmeans.cluster_centers_[team].astype(int)
    print(f"Team {team} (BGR: {center}): {[int(j) for j in players]}")

# ── Heatmaps ──────────────────────────────────────────────

print("\nGenerating heatmaps...")
os.makedirs("heatmaps", exist_ok=True)

# Individual
for tid, positions in player_history.items():
    if len(positions) < 20:
        continue
# By Team
generate_heatmap(f"Player #{tid}", positions,
                    f"heatmaps/player_{tid:03d}.png")

team_colormaps = {0: "Reds", 1: "Blues"}
for team, players in teams.items():
    positions = [p for tid in players for p in player_history[tid]]
    if len(positions) < 20:
        continue
    generate_heatmap(f"Team {team}", positions,
                    f"heatmaps/team_{team}.png",
                    cmap=team_colormaps[team])
    print(f"  team {team} → heatmaps/team_{team}.png")

# Global
all_positions = [p for positions in player_history.values() for p in positions]
generate_heatmap("Global", all_positions, "heatmaps/global.png")
print("  global → heatmaps/global.png")

# ── Second Pass: Annotated Video with Teams ─────────────

print("\nSecond pass — generating annotated video...")
TEAM_COLOR = {
    0: (50,  50,  200),   # red
    1: (200, 200, 200),   # white
   -1: (128, 128, 128),   # gray for unclassified players
}

tracker2 = sv.ByteTrack()
video_capture_2 = cv2.VideoCapture(VIDEO)
width = int(video_capture_2.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture_2.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_writer = cv2.VideoWriter(
    "output_teams.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    frames_per_second, (width, height)
)

while video_capture_2.isOpened():
    success, current_frame = video_capture_2.read()
    if not success:
        break

    detection_results = model(current_frame, conf=0.3)[0]
    detections_obj = sv.Detections.from_ultralytics(detection_results)
    detections_obj = detections_obj[np.isin(detections_obj.class_id, [1, 2])]
    detections_obj = tracker2.update_with_detections(detections_obj)

    if detections_obj.tracker_id is not None:
        for i, tid in enumerate(detections_obj.tracker_id):
            x1, y1, x2, y2 = map(int, detections_obj.xyxy[i])
            team = team_assignment.get(int(tid), -1)
            color = TEAM_COLOR[team]
            cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(current_frame, f"#{tid} T{team}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

    video_writer.write(current_frame)
    cv2.imshow("Teams", current_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture_2.release()
video_writer.release()
cv2.destroyAllWindows()

print("\nDone:")
print("  - heatmaps/        → individual and team heatmaps")
print("  - output_teams.mp4 → annotated video with teams")