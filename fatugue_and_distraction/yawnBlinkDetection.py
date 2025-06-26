import queue  # Used for thread-safe frame buffering
import threading  # Handles video capture and processing in parallel
import time
import winsound
import cv2
import numpy as np
from ultralytics import YOLO # Import YOLO for object detection 
import sys
from thresholds import *  # Import thresholds for blink and yawn detection
import time
import torch

# Global Variables for GUI/or live streaming 
num_of_blinks_gui = 0
microsleep_duration_gui = 0
num_of_yawns_gui = 0
yawn_duration_gui = 0
blinks_per_minute_gui = 0
yawns_per_minute_gui = 0



class DrowsinessDetector(): 
    def __init__(self):
        super().__init__()

        # Store current states
        self.yawn_state = ''
        self.eyes_state = ''
        self.alert_text = ''

        # Track statistics
        self.num_of_blinks = 0
        self.microsleep_duration = 0
        self.num_of_yawns = 0
        self.yawn_duration = 0

        # Track blinks/yawns per minute
        self.blinks_per_minute = 0
        self.yawns_per_minute = 0
        self.current_blinks = 0
        self.current_yawns = 0
        self.time_window = 60  # 1-minute window
        self.start_time = time.time()  # Track start time

        # Simplified blink and microsleep detection
        self.eyes_closed = False
        self.eyes_closed_start_time = None
        self.blink_threshold = 0.5  # Duration threshold to distinguish blink from microsleep
        self.last_blink_time = 0
        self.min_blink_interval = 0.2  # Minimum time between blinks
        
        # Yawn detection with moderate filtering
        self.yawn_confidence_threshold = 0.6  # Moderate confidence threshold
        self.consecutive_yawn_frames = 0
        self.min_yawn_frames = 3  # Require 3 consecutive frames
        self.yawn_detection_history = []
        self.max_history_length = 5
      
        # Initialize yawn-related tracking variables
        self.yawn_finished = False
        self.yawn_in_progress = False

        # Store the latest frame globally within the class
        self.current_frame = None  

        # Use CUDA if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        if torch.cuda.is_available():
            print("[INFO] CUDA is available. Using GPU for faster conversion.")
        else:
            print("[INFO] CUDA is not available. Using CPU for conversion.")

        # Load YOLO model
        # put you trained weights here in this path .
        # r"yolov5s.pt" is a placeholder, replace it with your actual model path.
        # Ensure the model is trained for drowsiness detection with appropriate classes and configurations. 
        self.detect_drowsiness = YOLO(r"")
        self.detect_drowsiness.to(self.device)  # Use selected device for inference
        
        # Using Multi-Threading (Only for tracking blink/yawn rates)
        self.stop_event = threading.Event()
        self.blink_yawn_thread = threading.Thread(target=self.update_blink_yawn_rate)
        self.blink_yawn_thread.start()  # Start the blink/yawn tracking thread

    def predict(self):
        """Processes the current frame and returns the detected state for eyes and yawning."""
        if self.current_frame is None:
            return "No Detection", 0.0

        results = self.detect_drowsiness(self.current_frame, device=self.device)
        if results is None or len(results) == 0 or results[0] is None:
            return "No Detection", 0.0

        if not hasattr(results[0], 'boxes') or results[0].boxes is None:
            return "No Detection", 0.0

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return "No Detection", 0.0

        confidences = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
        class_ids = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []
        
        if len(confidences) == 0 or len(class_ids) == 0:
            return "No Detection", 0.0

        max_confidence_index = np.argmax(confidences)
        class_id = int(class_ids[max_confidence_index])
        confidence = confidences[max_confidence_index]

        # Return classification based on class_id
        if class_id == 0:
            return "Opened Eye", confidence
        elif class_id == 1:
            return "Closed Eye", confidence
        elif class_id == 2:
            return "Yawning", confidence
        else:
            return "No Yawn", confidence

    def is_yawn_reliable(self, confidence):
        """Simple yawn reliability check"""
        self.yawn_detection_history.append(confidence)
        if len(self.yawn_detection_history) > self.max_history_length:
            self.yawn_detection_history.pop(0)
        
        # Check if most recent detections are high confidence
        high_confidence_count = sum(1 for conf in self.yawn_detection_history if conf > self.yawn_confidence_threshold)
        return high_confidence_count >= len(self.yawn_detection_history) * 0.6

    def process_frames(self, frame):
        """Receives and stores the latest frame, then processes it for detection."""
        global num_of_blinks_gui, microsleep_duration_gui, num_of_yawns_gui
        global yawn_duration_gui, blinks_per_minute_gui, yawns_per_minute_gui

        # Store the latest frame globally inside the class
        self.current_frame = frame  

        try:
            self.eyes_state, confidence = self.predict()
            current_time = time.perf_counter()

            # SIMPLIFIED: Eye blink and microsleep detection
            if self.eyes_state == "Closed Eye":
                if not self.eyes_closed:
                    # Eyes just closed - start tracking
                    self.eyes_closed = True
                    self.eyes_closed_start_time = current_time
                    print(f"👁️ Eyes closed at {current_time:.2f}")
                
                # Calculate duration eyes have been closed
                if self.eyes_closed_start_time:
                    eyes_closed_duration = current_time - self.eyes_closed_start_time
                    
                    # Update microsleep duration if longer than blink threshold
                    if eyes_closed_duration > self.blink_threshold:
                        self.microsleep_duration = eyes_closed_duration
                        microsleep_duration_gui = self.microsleep_duration
                        print(f"🔄 Microsleep duration: {self.microsleep_duration:.2f}s")
                
            else:
                # Eyes are open
                if self.eyes_closed:
                    # Eyes just opened - determine if it was a blink or microsleep
                    if self.eyes_closed_start_time:
                        eyes_closed_duration = current_time - self.eyes_closed_start_time
                        
                        if eyes_closed_duration <= self.blink_threshold:
                            # This was a blink
                            if current_time - self.last_blink_time >= self.min_blink_interval:
                                self.num_of_blinks += 1
                                num_of_blinks_gui = self.num_of_blinks
                                self.current_blinks += 1
                                self.last_blink_time = current_time
                                print(f"👁️ Blink detected! Total: {self.num_of_blinks}, Duration: {eyes_closed_duration:.2f}s")
                        else:
                            # This was a microsleep
                            print(f"⚠️ Microsleep ended! Duration: {eyes_closed_duration:.2f}s")
                        
                        # Reset microsleep duration
                        self.microsleep_duration = 0
                        microsleep_duration_gui = self.microsleep_duration
                    
                    # Reset closed eye state
                    self.eyes_closed = False
                    self.eyes_closed_start_time = None

            # SIMPLIFIED: Yawn detection
            if self.eyes_state == "Yawning" and confidence > self.yawn_confidence_threshold:
                if self.is_yawn_reliable(confidence):
                    self.consecutive_yawn_frames += 1
                    
                    if not self.yawn_in_progress and self.consecutive_yawn_frames >= self.min_yawn_frames:
                        # Start tracking yawn
                        self.start = time.perf_counter()
                        self.yawn_in_progress = True
                        self.yawn_duration = 0
                        print(f"🟡 Yawn started! Confidence: {confidence:.2f}")
                    
                    if self.yawn_in_progress:
                        self.yawn_duration = time.perf_counter() - self.start
                        yawn_duration_gui = self.yawn_duration

                        if yawn_duration_gui > yawning_threshold and not self.yawn_finished:
                            self.yawn_finished = True
                            self.num_of_yawns += 1
                            num_of_yawns_gui = self.num_of_yawns
                            self.current_yawns += 1
                            print(f"Yawn detected! Total: {self.num_of_yawns}, Duration: {self.yawn_duration:.2f}s")
                else:
                    # Reset if not reliable
                    self.consecutive_yawn_frames = 0
                    if self.yawn_in_progress:
                        self.yawn_in_progress = False
                        self.yawn_finished = False
                        self.yawn_duration = 0
                        yawn_duration_gui = self.yawn_duration

            else:
                # Reset yawn tracking when not yawning
                self.consecutive_yawn_frames = 0
                if self.yawn_in_progress:
                    self.yawn_in_progress = False
                    self.yawn_finished = False

                self.yawn_duration = 0
                yawn_duration_gui = self.yawn_duration

        except Exception as e:
            print(f"Error in processing the frame: {e}") 

    def update_blink_yawn_rate(self):
        """Updates blink and yawn rates every minute."""
        global blinks_per_minute_gui, yawns_per_minute_gui

        while not self.stop_event.is_set():
            time.sleep(self.time_window)
            self.blinks_per_minute = self.current_blinks
            blinks_per_minute_gui = self.blinks_per_minute
            self.yawns_per_minute = self.current_yawns
            yawns_per_minute_gui = self.yawns_per_minute

            print(f"Updated Rates - Blinks: {self.blinks_per_minute} per min, Yawns: {self.yawns_per_minute} per min")

            # Reset for next cycle
            self.current_blinks = 0
            self.current_yawns = 0

    def fatigue_detection(self):
        """Triggers alerts based on fatigue detection using the latest frame."""
        if self.current_frame is None:
            return

        microsleep_duration = microsleep_duration_gui
        blink_rate = blinks_per_minute_gui
        yawning_rate = yawns_per_minute_gui

    def play_alert_sound(self):
        """Plays an alert sound for fatigue detection."""
        frequency = 1000
        duration = 500
        winsound.Beep(frequency, duration)

    def play_sound_in_thread(self):
        """Runs the alert sound in a separate thread."""
        sound_thread = threading.Thread(target=self.play_alert_sound)
        sound_thread.start() 