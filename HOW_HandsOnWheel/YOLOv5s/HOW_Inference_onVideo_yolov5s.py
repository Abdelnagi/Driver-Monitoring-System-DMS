import pathlib
import torch
import cv2
import numpy as np

# 🩹 Patch for Windows compatibility
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model using ultralytics
# put your best.pt trained weights in path=''
model = torch.hub.load('ultralytics/yolov5', 'custom', path='', trust_repo=True, force_reload=True)
model.conf = 0.25
model.iou = 0.45

# Your custom classes
#-----------------------------------------------------
# DON'T FORGET TO CHECK YOUR data.yaml FOR CLASS INDEX
#-----------------------------------------------------
CLASSES = ['Hands On Wheel', 'Hands Off Wheel']
COLORS = [(0, 255, 0), (0, 0, 255)]  # Green, Red

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Load video file path here you want to infer on 
video_path = r""
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), f"Could not open video file: {video_path}"

print("[INFO] Running inference on video... Press ESC to exit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video stream.")
        break

    # Convert to RGB and run inference
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb, size=640)
    detections = results.xyxy[0].cpu().numpy()

    # Draw bounding boxes
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)
        label = f"{CLASSES[cls_id]} {conf:.2f}"
        color = COLORS[cls_id]

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Resize frame for smaller display
    resized_frame = cv2.resize(frame, (720, 480))

    # Display frame
    cv2.imshow("Detection", resized_frame)
    if cv2.waitKey(10) == 27:  # Faster playback
        break

cap.release()
cv2.destroyAllWindows()
