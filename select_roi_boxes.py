import cv2
import json

VIDEO_PATH = "example_video.mp4"   
OUTPUT_FILE = "roi_boxes.json"     

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read video")

rois = cv2.selectROIs("Select ROIs", frame, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

roi_boxes = [tuple(map(int, r)) for r in rois] 

print(f"Selected {len(roi_boxes)} ROIs:")
for i, (x, y, w, h) in enumerate(roi_boxes):
    print(f"  {i}: (x={x}, y={y}, w={w}, h={h})")

with open(OUTPUT_FILE, "w") as f:
    json.dump(roi_boxes, f, indent=2)

