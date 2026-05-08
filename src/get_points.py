import cv2
import argparse
import os

parser = argparse.ArgumentParser(description="Select reference points for homography.")
parser.add_argument("--video", type=str, default="../videos/Untitled design.mp4", help="Video path.")
args = parser.parse_args()

video_path = args.video

if not os.path.exists(video_path):
    print(f"Error: File '{video_path}' does not exist.")
    exit()

selected_points = []
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])
        cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
        cv2.putText(frame, str(len(selected_points)), (x+10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imshow("Select 4 field corners", frame)
        print(f"Point {len(selected_points)}: ({x}, {y})")

video_capture = cv2.VideoCapture(video_path)
ret, frame = video_capture.read()
video_capture.release()

if not ret or frame is None:
    print(f"Error: The frame video could not be read from '{video_path}'.")
    exit()

cv2.namedWindow("Select 4 field corners", cv2.WINDOW_FULLSCREEN)
cv2.imshow("Select 4 field corners", frame)
cv2.setMouseCallback("Select 4 field corners", on_mouse_click)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Your points:", selected_points)