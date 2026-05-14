# Football CV — Player Tracking & Analysis

Computer vision system for football tactical analysis from broadcast video. It detects players, tracks them throughout the match, classifies them by team, and generates positioning heatmaps — without the need for specialized hardware.

---

## Demo

| Annotated video with teams | Heatmap by team |
|---|---|
| Bounding boxes by team with persistent IDs | Presence zones on the actual pitch |

---

## Pipeline

```
Broadcast video
    ↓
Detection (YOLOv8 fine-tuned)
    ↓
Multi-object tracking (ByteTrack)
    ↓
Re-identification (ResNet50 embeddings)
    ↓
Pitch projection (OpenCV homography)
    ↓
Team classification (K-means on jersey color)
    ↓
Heatmaps + Annotated video
```

---

## Features

 - **Specialized Detection** — Fine-tuned YOLOv8 using the Roboflow football dataset. Detects players, goalkeepers, referees, and the ball as separate classes. mAP50: 84.7%.
 - **Persistent Tracking** — ByteTrack assigns a unique ID to each player and maintains it throughout the video even with partial occlusions.
 - **Visual Re-identification** — When the tracker loses a player and re-detects them, a pretrained ResNet50 extracts visual embeddings and compares them with the gallery of known players to reassign the original ID instead of creating a new one.
 - **Homography** — Transforms video pixel coordinates to real FIFA pitch meters (105×68m) using reference points from the pitch lines.
 - **Team Classification** — K-means on the average torso color of each player (with a grass green mask) automatically separates the two teams unsupervised.
 - **Heatmaps** — Individual, team, and global positioning density maps on the 2D pitch.

---

## Tech Stack

- Python 3.10+
- [YOLOv8](https://github.com/ultralytics/ultralytics) — object detection
- [Supervision](https://github.com/roboflow/supervision) — ByteTrack + annotation
- [OpenCV](https://opencv.org/) — homography and video processing
- [PyTorch + torchvision](https://pytorch.org/) — ResNet50 backbone for re-id
- [scikit-learn](https://scikit-learn.org/) — K-means for team classification
- [matplotlib + scipy](https://matplotlib.org/) — heatmap generation

---

## Installation

```bash
git clone https://github.com/belenarbizu/computer_vision_football
cd football-cv
pip install -r requirements.txt
```

**requirements.txt**
```
ultralytics
supervision
opencv-python
torch
torchvision
scikit-learn
matplotlib
scipy
numpy
```

---

## Usage

### 1. Select pitch reference points

```bash
python get_points.py --video your_video.mp4
```

Click on 4 points on the pitch lines for which you know the coordinates in meters (penalty box corners, center line, etc.) and note down the pixel values it prints.

### 2. Execute complete pipeline

```python
# Edit theses variables in main.py
src_points = np.array([...])  # pixels of the video
dst_points = np.array([...])  # reals coordinates in meters
```

```bash
python main.py --video your_video.mp4
```

### 3. Generated outputs

```
heatmaps/
├── player_001.png   # individual heatmap per player
├── player_002.png
├── ...
├── team_0.png      # aggregated heatmap of team 0
├── team_1.png      # aggregated heatmap of team 1
└── global.png        # heatmap of all players

team_outputs.mp4    # video with annotated players and teams
```

---

## Technical decisions

**Why fine-tune YOLOv8 instead of using a generic detector?**

Base YOLOv8 trained on COCO classifies everything as person, with no distinction between players and referees and poor ball detection. Fine-tuning on the Roboflow Football dataset adds the classes player, goalkeeper, referee and ball, enabling referees to be filtered from the analysis and goalkeepers to be handled separately. The result is a mAP50 of 84.7% compared to ~40% for the base model on this domain.

**Why ByteTrack instead of DeepSORT?**

DeepSORT requires visual features on every frame for association, which increases latency. ByteTrack associates detections using only IoU and a lost-track buffer, making it faster and more robust with partial occlusions. Visual re-identification is applied as an additional layer on top of ByteTrack rather than inside the tracker, keeping the system modular.

**Why ResNet50 for re-identification instead of a trained Siamese Network?**

A Siamese Network trained specifically on football players would yield better results, but requires a dataset labelled with player identities across multiple frames — data that is not publicly available. Pre-trained ResNet50 on ImageNet extracts visual embeddings discriminative enough to reduce unique IDs by ~40% compared to ByteTrack alone (from ~53 to 32 IDs on a 30-second clip). The natural next step would be training a re-id network on domain-specific data using Triplet Loss.

**Why K-means for team classification?**

This is an unsupervised clustering problem with exactly 2 known clusters. K-means on the mean BGR torso color (with a green pitch mask in HSV space) is robust enough for teams with clearly different colors. Limitations arise when team colors are similar or when there is significant lighting variation.

---

## Known limitations

- **Camera movement** — The homography is computed with fixed reference points. When the camera pans or zooms, pitch projections introduce error. The solution is to automatically recalibrate the homography each frame by detecting pitch lines.
- **Strong occlusions** — When several players fully overlap for multiple frames, the tracker loses their IDs and re-identification does not always recover them.
- **Ball in motion** — Motion blur reduces ball detection to ~25% of frames when the ball is moving fast. This would require a specialized model or temporal interpolation.
- **Similar team colors** — If both teams wear similar colors, K-means may misclassify some players. This could be improved with semantic jersey segmentation.

---

## Dataset

- **Detection:** [Football Players Detection — Roboflow Universe](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- **Test videos:** [DFL Bundesliga — Kaggle](https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv)

---

## Model results

| Class | mAP50 |
|---|---|
| player | 80.8% |
| goalkeeper | 72.6% |
| referee | 67.1% |
| ball | 25.2% |
| **overall** | **84.7%** |

Trained with YOLOv8n, 50 epochs, imgsz=1280, AdamW optimizer, T4 GPU (Google Colab).

---

## Roadmap

- Automatic per-frame homography recalibration using pitch line detection
- Siamese/Triplet Network trained on domain-specific data for more robust re-id
- Storage of player metrics in a PostgreSQL database
- Interactive dashboard with Streamlit or Plotly

---

## Author

Developed as a portfolio project for Computer Vision & Machine Learning Engineer positions in sports analytics.

---

Siamese networks:
https://medium.com/@rinkinag24/a-comprehensive-guide-to-siamese-neural-networks-3358658c0513

Si ByteTrack asigna un ID que ya existía → actualizar su galería normalmente
Si ByteTrack asigna un ID nuevo → buscar en la galería si es un jugador que ya habíamos visto → si coincide, reasignar el ID antiguo
