import pathlib
import sys
pathlib.PosixPath = pathlib.WindowsPath

import threading
import queue
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from collections import Counter
from PIL import Image
import sys
import json
from collections import deque
from pathlib import Path
import os

class CustomModel(nn.Module):
    def __init__(self, model_path, labels, classes, device="CPU"):
        super(CustomModel, self).__init__()
        # Use CUDA if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        if torch.cuda.is_available():
            print("[INFO] CUDA is available. Using GPU for faster conversion.")
        else:
            print("[INFO] CUDA is not available. Using CPU for conversion.")

        try:
            self.model = self.load_model(model_path, len(classes))
            self.model.eval()
            self.labels = labels
            self.classes = classes
            self.model = self.model.to(self.device)
            print("Model loaded and moved to device successfully")
        except Exception as e:
            print(f"Error in CustomModel initialization: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def load_model(self, model_path, num_classes):
        print("Creating base MobileNetV3 model...")
        model = models.mobilenet_v3_large(pretrained=False)
        print(f"Modifying classifier for {num_classes} classes...")
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        print(f"Loading state dict from {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=torch.device(self.device))
            model.load_state_dict(state_dict)
            print("State dict loaded successfully")
        except Exception as e:
            print(f"Error loading state dict: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        return model

class AD_HOW_Detection:
    def __init__(self):
        # Initialize paths and constants
        self.JSON_PATH = "driver_assistant.json"
        self.input_video_fps = None
        
        # Global adjustable thresholds
        self.CONF_THRESHOLD = 0.5
        self.IOU_THRESHOLD = 0.45

        # Initialize device and models
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU instead.")
        
        self.model_path = r"D:\fatigue\fine_tuned_mobilenetv3_with_aug_benchmark.pth"
        print(f"AD model path: {self.model_path}")
        
        # Initialize labels and class mappings
        self.labels = list(range(0, 10))
        self.class_labels = {
            0: "Safe driving",
            1: "Texting(right hand)",
            2: "Talking on the phone (right hand)",
            3: "Texting (left hand)",
            4: "Talking on the phone (left hand)",
            5: "Operating the radio",
            6: "Drinking",
            7: "Reaching behind",
            8: "Hair and makeup",
            9: "Talking to passenger(s)",
        }
        
        # Verify model file exists
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model file not found at {self.model_path}")
            self.ad_model = None
        else:
            print("Model file found, attempting to load...")
            try:
                self.ad_model = CustomModel(self.model_path, self.labels, self.class_labels, device=self.device).model
                print("AD model loaded successfully")
            except Exception as e:
                print(f"Error loading AD model: {str(e)}")
                import traceback
                traceback.print_exc()
                self.ad_model = None

        # Initialize YOLO model
        self.weights_path = r"D:\fatigue\best.pt"
        self.model_HOW = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weights_path, device=self.device, force_reload=True)
        self.model_HOW.conf = self.CONF_THRESHOLD  # Set confidence threshold
        self.model_HOW.iou = self.IOU_THRESHOLD    # Set IoU threshold

        # Initialize queues and events
        self.frame_queue_AD = queue.Queue(maxsize=5)
        self.frame_queue_HOW = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()

        # Initialize status variables
        self.per_frame_driver_activity = "Unknown (0.0%)"
        self.per_frame_hands_on_wheel = "No (0.00)"
        self.driver_state = "N/A"
        self.confidence_text = "N/A"
        self.hands_state = "N/A"
        self.hands_confidence = "N/A"
        self.latest_frame = None
        self.majority_class = "N/A"
        # ➕ Add these two lines here:
        self.latest_ad_label = "Unknown"
        self.latest_ad_conf = "0.00"

        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_HOW(self, frame):
        # YOLOv5 handles preprocessing internally
        return frame

    def capture_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.input_video_fps = fps if fps > 0 else 30
        frame_time = 1 / self.input_video_fps


        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if frame is None:
                print("No frame available, waiting...")
                time.sleep(frame_time)
                continue
            if ret:
                if self.frame_queue_AD.full():
                    self.frame_queue_AD.get()
                if self.frame_queue_HOW.full():
                    self.frame_queue_HOW.get()
            self.frame_queue_AD.put(frame.copy())
            self.frame_queue_HOW.put(frame.copy())
            #print(f"Added frame to queues - AD size: {self.frame_queue_AD.qsize()}, HOW size: {self.frame_queue_HOW.qsize()}")
            time.sleep(frame_time)

    def predict_HOW(self, frame):
        # Run YOLOv5 inference
        results = self.model_HOW(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size=640)
        detections = results.xyxy[0].cpu().numpy()

        if len(detections) == 0:
            print("No HOW detection.")
            return None, 0.0, None

        # Define class labels and colors
        CLASSES = ['Hands On Wheel', 'Hands Off Wheel']
        COLORS = [(0, 255, 0), (0, 0, 255)]  # Green for On, Red for Off

        highest_conf = 0
        highest_cls = None
        best_box = None

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)

            # Sanity check for class ID
            if cls_id < 0 or cls_id >= len(CLASSES):
                continue

            label = f"{CLASSES[cls_id]} {conf:.2f}"
            color = COLORS[cls_id]

            # Draw the bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Track highest confidence detectionF
            if conf > highest_conf:
                highest_conf = conf
                highest_cls = cls_id
                best_box = [x1, y1, x2, y2]

        return highest_cls, highest_conf, best_box


    def process_frames_HOW(self):
        writer = None
        output_path = "full_detection_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.input_video_fps if self.input_video_fps else 30


        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue_HOW.get()
                highest_cls, highest_conf, best_box = self.predict_HOW(frame)

                # ----- Overlay Per-frame AD label -----
                ad_text = f"AD: {self.latest_ad_label} ({self.latest_ad_conf}%)"
                cv2.putText(frame, ad_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

                # ----- Overlay Majority AD class -----
                maj_ad_text = f"🧠 Majority AD: {self.majority_class}"
                cv2.putText(frame, maj_ad_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # ----- Overlay Majority HOW state -----
                maj_how_text = f"👐 Majority HOW: {self.hands_state}"
                color = (0, 255, 0) if self.hands_state == "Hands On Wheel" else (0, 0, 255)
                cv2.putText(frame, maj_how_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # ----- Init and write video -----
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    print(f"[INFO] Video writer initialized: {output_path} (FPS={fps})")

                writer.write(frame)

                if self.result_queue.qsize() < 100:
                    self.result_queue.put(("HOW", frame, (highest_cls, highest_conf, best_box)))
                else:
                    time.sleep(0.1)
                    print("Processing batch of 100 frames (HOW)")
                    while not self.result_queue.empty():
                        self.result_queue.get()

            except Exception as e:
                print(f"Error in HOW: {e}")

        if writer is not None:
            writer.release()
            print("[INFO] Full annotated video saved.")


    def predict_activity_AD(self, frame, model):
        try:
            if frame is None:
                print("Error: Received None frame in predict_activity_AD")
                return [("Unknown", 0.0)]
                
            if model is None:
                print("Error: Model is None in predict_activity_AD")
                return [("Unknown", 0.0)]
                
            # Convert frame to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            
            # Apply transformations
            img = self.transform(img).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(img)
                probabilities = F.softmax(outputs, dim=1)
                top_prob, top_index = torch.topk(probabilities, 1)
                top_label = [self.class_labels[idx.item()] for idx in top_index[0]]
                top_confidence = [round(prob.item() * 100, 2) for prob in top_prob[0]]
                
                print(f"AD prediction successful - Label: {top_label[0]}, Confidence: {top_confidence[0]}%")
                return list(zip(top_label, top_confidence))
                
        except Exception as e:
            print(f"Error in activity prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return [("Unknown", 0.0)]

    def process_frames_AD(self):
        if self.ad_model is None:
            print("AD model not initialized, skipping processing")
            return
            
        while not self.stop_event.is_set():
            try:
                print(f"AD proooooooooooceeeeeeeeeeeeesS")
                frame = self.frame_queue_AD.get()
                print("Processing AD frame...")
                top_prediction = self.predict_activity_AD(frame, self.ad_model)
                if top_prediction:
                    self.latest_ad_label, self.latest_ad_conf = top_prediction[0]


                print(f"AD prediction: {top_prediction}")
                if self.result_queue.qsize() < 100:
                    self.result_queue.put(("AD", frame, top_prediction))
                else:
                    time.sleep(0.1)
                    print("Processing batch of 100 frames (AD)")
                    while not self.result_queue.empty():
                        self.result_queue.get()
                print(f"AD prediction added to queue, new size: {self.result_queue.qsize()}")
            except Exception as e:
                print(f"Error in activity prediction: {e}")

    def majority_how_update(self):
        queue_list = list(self.result_queue.queue)
        how_predictions = [predictions for source, _, predictions in queue_list if source == "HOW"]

        if len(how_predictions) < 25:
            return

        hands_on_counter = 0
        hands_off_counter = 0

        for predictions in how_predictions:
            detected_label = predictions[0]
            #print(f"detected_label in majority function is: { detected_label}")
            if detected_label is None:
                continue
            if detected_label == 0:
                hands_on_counter += 1
            elif detected_label == 1:
                hands_off_counter += 1

        if hands_on_counter > hands_off_counter:
            self.hands_state = "Hands On Wheel"
            self.hands_confidence = "✅Driver is in control"
        else:
            self.hands_state = "Hands Off Wheel"
            self.hands_confidence = "⚠🚨WARNING! Hands off wheel detected!"
        print(f"HANDs Majority Updated: {self.hands_state}, {self.hands_confidence}")

    def majority_driving_state_update(self):
        queue_list = list(self.result_queue.queue)
        ad_predictions = [predictions for source, _, predictions in queue_list if source == "AD"]
        
        print(f"Current queue size: {self.result_queue.qsize()}, AD frame count: {len(ad_predictions)}")

        if (self.result_queue.qsize()) < 100:
            print(f"Queue size is {self.result_queue.qsize()}, waiting for 100 AD frames...")
            return

        safe_counter = 0
        unsafe_counter = 0
        for predictions in ad_predictions:
            driver_label, _ = predictions[0]
            if driver_label == "Safe driving":
                safe_counter += 1
            else:
                unsafe_counter += 1
        if safe_counter > unsafe_counter:
            self.driver_state = "✅Safe driving"
            self.confidence_text = "Good job,you're driving safely"
        else:
            self.driver_state = "❌Unsafe driving"
            self.confidence_text = "⚠🚨ALERT!!! PAY ATTENTION TO THE ROAD"
        if self.hands_state == "Hands Off Wheel":
            self.driver_state = "❌Unsafe driving"
            self.confidence_text = "⚠🚨ALERT!!! PUT YOUR HANDS ON THE WHEEL"
        print(f"AD Majority Updated: {self.driver_state}, {self.confidence_text}")

    def majority_AD_update(self):
        queue_list = list(self.result_queue.queue)
        ad_predictions = [predictions for source, _, predictions in queue_list if source == "AD"]

        if (self.result_queue.qsize()) < 100:
            print(f"Queue size is {self.result_queue.qsize()}, waiting for 100 AD frames...")
            return

        class_counts = [0] * 10
        
        for predictions in ad_predictions[-100:]:
            driver_label, _ = predictions[0]
            for class_idx, class_name in self.class_labels.items():
                if driver_label == class_name:
                    class_counts[class_idx] += 1
                    break
        
        max_count = max(class_counts)
        majority_class_idx = class_counts.index(max_count)
        self.majority_class = self.class_labels[majority_class_idx]

    def update_driver_assistant_field(self, **field_updates):
        try:
            # Initialize with default values if file doesn't exist or is empty
            default_data = {
                "Activity_Alert": "Unknown",
                "HOW_Alert": "Unknown",
                "last_update": ""
            }
            
            if os.path.exists(self.JSON_PATH):
                try:
                    with open(self.JSON_PATH, "r") as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    # If file is empty or invalid, use default data
                    data = default_data
            else:
                data = default_data

            # Update the data with new values
            data.update(field_updates)
            data["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")

            # Write the updated data back to file
            with open(self.JSON_PATH, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error updating driver assistant fields: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_status_loop(self):
        while not self.stop_event.is_set():
            try:
                queue_list = list(self.result_queue.queue)
                driver_state_gui = "Unknown"
                conf_gui = "N/A"
                highest_cls = "N/A"
                highest_conf = 0.0

                for source, frame, prediction in queue_list[-2:]:
                    if frame is not None:
                        self.latest_frame = frame
                    if source == "AD":
                        driver_state_gui, conf_gui = prediction[0]
                    elif source == "HOW":
                        highest_cls, highest_conf, best_box = prediction

                self.per_frame_driver_activity = f"{driver_state_gui} ({conf_gui}%)"
                if highest_cls == 0:
                    yes_no = "Hands On Wheel"
                elif highest_cls == 1:
                    yes_no = "Hands Off Wheel"
                else:
                    yes_no = "Unknown"

                self.per_frame_hands_on_wheel = f"{yes_no} ({highest_conf:.2f})"
                self.majority_driving_state_update()
                self.majority_how_update()
                self.majority_AD_update()

                self.update_driver_assistant_field(
                    Activity_Alert=self.majority_class,
                    HOW_Alert=self.hands_state,
                )

                data = {
                    "per_frame_driver_activity": self.per_frame_driver_activity,
                    "per_frame_hands_on_wheel": self.per_frame_hands_on_wheel,
                    "majority_driver_state": self.driver_state,
                    "system_alert": self.confidence_text,
                    "hands_monitoring": self.hands_state,
                    "hands_monitoring_confidence": self.hands_confidence
                }
                with open("status.json", "w") as f:
                    json.dump(data, f)
            except Exception as e:
                print(f"Error in update_status_loop: {e}")
            time.sleep(0.1)

    def start_detection(self, video_path):
        start_time = time.time()

        # Start threads
        capture_thread = threading.Thread(target=self.capture_frames, args=(video_path,))
        ad_thread = threading.Thread(target=self.process_frames_AD)
        how_thread = threading.Thread(target=self.process_frames_HOW)
        status_thread = threading.Thread(target=self.update_status_loop)

        capture_thread.start()
        ad_thread.start()
        how_thread.start()
        status_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Interrupt received, stopping threads...")
            self.stop_event.set()
            capture_thread.join()
            ad_thread.join()
            how_thread.join()
            status_thread.join()

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    video_path = input("Enter the video file path (or press Enter to use the live feed URL): ")
    if not video_path:
        video_path="http://127.0.0.1:5000/video_feed1"
    else:
       print(f"Using video file: {video_path}")

    detector = AD_HOW_Detection()
    detector.start_detection(video_path)
