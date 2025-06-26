
import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
import threading
from queue import Queue



# Configuration
#  - put you trained weights in CUSTOM_WEIGHTS_PATH which is trained with yolov7
#  - put youl yolov7 repo path in YOLOV7_REPO_PATH
#  - put the video path you want to infer on it 
#
CUSTOM_WEIGHTS_PATH = r""  
YOLOV7_REPO_PATH = r""  
VIDEO_PATH = r""  
CONFIDENCE_THRESHOLD = 0.6  # Confidence threshold for detections


# Check if weights file exists
if not os.path.exists(CUSTOM_WEIGHTS_PATH):
    raise FileNotFoundError(f"Custom weights file not found: {CUSTOM_WEIGHTS_PATH}")

# Check if video file exists
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

# Add YOLOv7 repo to the Python path
sys.path.append(YOLOV7_REPO_PATH)

# Now we can import the necessary modules from YOLOv7
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Initialize device (GPU or CPU)
device = select_device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLOv7 model
model = torch.hub.load(YOLOV7_REPO_PATH, 'custom', CUSTOM_WEIGHTS_PATH, source='local', force_reload=True).to(device)
model.eval()

if device.type != 'cpu':
    model.half()  # Convert model to FP16


# Open video capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise Exception("Error: Could not open video file.")


# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no more frames
    
    original_frame = frame.copy()
    
    # Preprocess the frame
    img_resized = cv2.resize(frame, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
    

# Set FPS to match the original video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Use a queue for multi-threaded frame reading
frame_queue = Queue(maxsize=5)

def read_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)

threading.Thread(target=read_frames, daemon=True).start()

frame_skip = 40 # Process every 2nd frame
frame_count = 0

while cap.isOpened():
    if frame_queue.empty():
        continue  # Wait until frames are available
    
    frame = frame_queue.get()
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip processing for performance
    
    # Preprocess the frame
    img_resized = cv2.resize(frame, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).contiguous().to(device).float() / 255.0
    if device.type != 'cpu':
        img_tensor = img_tensor.half()  


    with torch.no_grad():
        predictions = model(img_tensor)
    

    # Extract detections
    detections = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
    detections = non_max_suppression(detections, CONFIDENCE_THRESHOLD, 0.45)[0]

    # Ensure predictions are in the correct format
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Extract detections
    detections = non_max_suppression(predictions, CONFIDENCE_THRESHOLD, 0.5)[0]

    
    # Initialize label
    label = "No Detection"
    color = (255, 255, 255)
    if detections is not None and len(detections):

        detections[:, :4] = scale_coords(img_tensor.shape[2:], detections[:, :4], original_frame.shape).round()

        detections[:, :4] = scale_coords(img_tensor.shape[2:], detections[:, :4], frame.shape).round()

        
        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            if conf < CONFIDENCE_THRESHOLD:
                continue
            
            #-----------------------------------------------------
            # DON'T FORGET TO CHECK YOUR data.yaml FOR CLASS INDEX
            #-----------------------------------------------------
            label = f"{'HandsOnWheel' if cls == 1 else 'HandsOffWheel'} {conf:.2f}"
            color = (255, 0, 0) if cls == 1 else (0, 0, 255)
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 2)
    
    # Display label
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    label_x = (original_frame.shape[1] - label_size[0]) // 2
    label_y = original_frame.shape[0] - 10
    cv2.putText(original_frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Show the frame with detections
    cv2.imshow("Real-Time Detection", original_frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):

            # notice that if the dataset is augmented you should check yaml and the labels
            # because roboflow flip all the labels
            # if augmented : handson -> 0 and handsoff -> 1
            # Determine label and color
            label = f"{'HandsOffWheel' if cls == 1 else 'HandsOnWheel'} {conf:.2f}"
            color = (0,0, 255) if cls == 1 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Display label
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Show the frame with detections
    cv2.imshow("Real-Time Detection", frame)
    
    # Maintain the original video FPS
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):

        break

# Release resources
cap.release()
cv2.destroyAllWindows()

cv2.destroyAllWindows()

