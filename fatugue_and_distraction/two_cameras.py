from flask import Flask, Response, render_template, jsonify
from waitress import serve
import threading
import cv2
import time
import json
import os
from flask import send_from_directory
import sys


app = Flask(__name__)

# ─────────────────────────────────────
# Multi-threaded Camera Capture Class
# ─────────────────────────────────────
class CameraStream:
    def __init__(self, source):
        # source can be either a camera ID (integer) or a video file path (string)
        self.source = source
        self.cap = None
        self.frame = None
        self.running = True
        self.last_frame_time = time.time()
        self.error_count = 0
        self.MAX_ERRORS = 3
        self.fps = 30  # Default FPS
        self.frame_interval = 1.0 / self.fps
        
        # Initialize camera with better error handling
        if not self.initialize_camera():
            raise ValueError(f"Failed to open video source: {source}")
            
        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

    def initialize_camera(self):
        try:
            if self.cap is not None:
                self.cap.release()
                time.sleep(0.1)  # Reduced sleep time
            
            # Check if source is a file path or camera ID
            if isinstance(self.source, str) and (self.source.endswith('.mp4') or self.source.endswith('.avi') or self.source.endswith('.mov')):
                # It's a video file
                self.cap = cv2.VideoCapture(self.source)
                # Get actual FPS from video file
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.fps = actual_fps if actual_fps > 0 else 30
                print(f"📹 Video file FPS: {self.fps}")
            else:
                # It's a camera device - try different backend APIs for faster initialization
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                for backend in backends:
                    try:
                        self.cap = cv2.VideoCapture(int(self.source), backend)
                        if self.cap.isOpened():
                            break
                        self.cap.release()
                    except Exception:
                        continue

            if not self.cap.isOpened():
                return False

            # Set camera properties only for actual cameras, not video files
            if not isinstance(self.source, str) or not (self.source.endswith('.mp4') or self.source.endswith('.avi') or self.source.endswith('.mov')):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS for cameras
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size for lower latency
                
                # Get actual FPS from camera
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.fps = actual_fps if actual_fps > 0 else 30
                print(f"📷 Camera {self.source} FPS: {self.fps}")
            
            # Update frame interval based on actual FPS
            self.frame_interval = 1.0 / self.fps
            
            # Read a test frame to ensure camera is working
            success, _ = self.cap.read()
            if not success:
                return False

            print(f"✅ Camera/Video {self.source} initialized successfully (FPS: {self.fps})")
            self.error_count = 0
            return True
        except Exception as e:
            print(f"⚠️ Error initializing camera/video {self.source}: {str(e)}")
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            return False

    def update_frames(self):
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    if self.error_count < self.MAX_ERRORS:
                        self.error_count += 1
                        print(f"⚠️ Camera/Video {self.source} lost connection, attempting to reinitialize ({self.error_count}/{self.MAX_ERRORS})")
                        if self.initialize_camera():
                            continue
                    time.sleep(0.5)  # Reduced sleep time
                    continue

                success, frame = self.cap.read()
                if success:
                    self.frame = frame
                    self.last_frame_time = time.time()
                    self.error_count = 0
                else:
                    # For video files, restart from beginning when reaching the end
                    if isinstance(self.source, str) and (self.source.endswith('.mp4') or self.source.endswith('.avi') or self.source.endswith('.mov')):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    current_time = time.time()
                    if current_time - self.last_frame_time > 2.0:  # Reduced timeout
                        print(f"⚠️ No frames from camera/video {self.source} for 2 seconds, reinitializing...")
                        self.initialize_camera()
                time.sleep(0.01)  # Reduced sleep time for faster response

            except Exception as e:
                print(f"⚠️ Error in camera/video {self.source} thread: {str(e)}")
                self.error_count += 1
                if self.error_count >= self.MAX_ERRORS:
                    print(f"❌ Too many errors for camera/video {self.source}, stopping attempts")
                    break
                time.sleep(0.5)  # Reduced sleep time

    def get_frame(self):
        if self.frame is None or time.time() - self.last_frame_time > 3.0:  # Reduced timeout
            return None
        return self.frame.copy()  # Return a copy to prevent race conditions
    
    def get_fps(self):
        return self.fps
    
    def get_frame_interval(self):
        return self.frame_interval
      
    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()

# ─────────────────────────────────────────────────────────────
# 🎥 REAL-TIME CAMERA INITIALIZATION (Front & Side Views)
# ─────────────────────────────────────────────────────────────
# This block initializes two camera feeds connected via USB ports.
# - camera1 → Side-view camera: used by the Activity and Hands-On-Wheel (HOW) module.
# - camera2 → Front-view camera: used by the Fatigue and Distraction Detection module.
# These are real-time streams from physical cameras connected to the laptop.
# This script is also attached to the Activity_HOW module for initialization.
print("🚀 Initializing cameras...")
camera1 = CameraStream(1)  # Side view
camera2 = CameraStream(0)  # Front view
print("✅ Both cameras initialized successfully")


# ─────────────────────────────────────────────────────────────
# 📼 OFFLINE VIDEO INPUT (Not Real-Time)
# ─────────────────────────────────────────────────────────────
# This section initializes both feeds from a recorded video file.
# It's used in cases where real-time cameras aren't available.
# The video was pre-captured from a real car scenario.
# NOTE: Make sure to set the path to the video file in `video_path`.
video_path = r""  # ← Add full path to your recorded video here

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30

VIDEO_FPS = get_video_fps(video_path)
FRAME_INTERVAL = 1.0 / VIDEO_FPS

camera1 = CameraStream(video_path)  # Simulated side view
camera2 = CameraStream(video_path)  # Simulated front view




# ───────────────────────────────
# Frame Generator with FPS Logging
# ───────────────────────────────
def generate_frames(camera_stream, name='cam'):
    prev_time = time.time()
    while True:
        start_time = time.time()
        frame = camera_stream.get_frame()
        if frame is None:
            continue

        # # Crop differently for camera2
        # if name == 'camera2 ':
        #     h, w, _ = frame.shape
        #     frame = frame[int(h*0.2):int(h*0.8), int(w*0.3):int(w*0.7)]  # Example center crop
        # Flip the frame horizontally (180 degrees)
        frame = cv2.flip(frame, 1)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ret:
          continue
        frame = buffer.tobytes()

        # Measure FPS and latency
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # in ms
        fps = 1 / (end_time - prev_time) if (end_time - prev_time) != 0 else 0
        prev_time = end_time
        
        # Get camera's actual FPS for comparison
        camera_fps = camera_stream.get_fps()
        print(f"[{name}] Actual FPS: {camera_fps:.1f} | Measured FPS: {fps:.2f} | Latency: {latency:.2f} ms")

        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        elapsed = time.time() - start_time
        sleep_time = max(0, camera_stream.get_frame_interval() - elapsed)
        time.sleep(sleep_time)

# ────────────────
# Flask Routes
# ────────────────

# Main page showing Camera 1 feed with status panel and Camera 2 feed below
@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>Driver Monitoring Live Stream</title>
        <!-- Responsive meta tag so the layout scales to device width -->
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        
        <style>
          /* Reset and ensure 100% height for html/body */
          html, body {
            margin: 0;
            padding: 0;
            
          }

          /* Make the gradient cover the entire visible area */
          body {
            min-height: 100vh;
            font-family: Arial, sans-serif;
            background: linear-gradient(to bottom, #7D33A3, #6882C5);
            background-repeat: no-repeat;
            background-size: cover;  /* Key: fill entire area */
            /* background-attachment: fixed;  <-- optional (often ignored on mobile) */
          }

          /* Flex container that can wrap to avoid horizontal scrolling */
          .container {
            display: flex;
            flex-wrap: wrap;        /* Allows cards to stack on small screens */
            justify-content: center;
            align-items: flex-start;
            gap: 40px;
            padding: 20px;
          }

          /* Card styling with a max-width for responsiveness */
          .card {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
            padding: 20px;
            width: 100%;           /* Take full width on narrow screens */
            max-width: 700px;      /* But don't exceed 700px on larger screens */
            box-sizing: border-box; /* Ensure padding doesn't overflow */
          }
          .card h2 {
            margin-top: 0;
          }

          /* Camera container and images: fully responsive */
          .camera-container {
            position: relative;
            width: 100%;
            max-width: 640px;   /* Same ratio as your feed's resolution */
            margin: 0 auto;     /* Center images within the card */
          }
          .camera-container img {
            display: block;
            width: 100%;        /* Scale down on small screens */
            height: auto;
          }
          .overlay {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
            width: 100%;        /* Match the underlying image width */
            height: auto;
          }

          /* Simple styling for status panels */
          .status-panel {
            margin-top: 20px;
            border: 1px solid #ccc;
            padding: 10px;
          }

          /* New confidence display box styling */
          .confidence-box {
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
          }
          .confidence-good {
            background-color: rgba(0, 255, 0, 0.2);
            color: green;
          }
          .confidence-bad {
            background-color: rgba(255, 0, 0, 0.2);
            color: red;
          }
        </style>
        
        <script>
          // Poll /status every 1 second
          function updateStatus() {
            fetch('/status')
              .then(response => response.json())
              .then(data => {
                // Update Camera 1 status panel
                document.getElementById("statusInfo").innerHTML = `
                  <div>
                    <strong style="font-size: 18px;">Activity and hands detection</strong>
                    <hr>
                    <div class="confidence-box ${data.camera1.system_alert === "Good job,you're driving safely" ? 'confidence-good' : 'confidence-bad'}">
                      ${data.camera1.system_alert}
                    </div>
                    <h3 style="margin: 0;">Per Frame Prediction</h3>
                    <p style="margin: 0;">
                      <b>Driver Activity:</b> ${data.camera1.per_frame_driver_activity}<br>
                      <b>Hands status:</b> ${data.camera1.per_frame_hands_on_wheel}
                    </p>
                    <hr>
                    <h3 style="margin: 0;">State Monitoring</h3>
                    <p style="margin: 0;">
                      <b>Majority Driver State:</b> ${data.camera1.majority_driver_state}<br>
                      <b>System Alert:</b> ${data.camera1.system_alert}<br>
                      <b>Hands Monitoring:</b> ${data.camera1.hands_monitoring}<br>
                      <b>Hands Monitoring Confidence:</b> ${data.camera1.hands_monitoring_confidence}
                    </p>
                  </div>
                `;
                
                // Update Camera 2 status panel
                document.getElementById("statusInfo2").innerHTML = `
                  <div>
                    <strong style="font-size: 18px;">Fatigue detection</strong>
                    <hr>
                    <div class="confidence-box ${data.camera2.alert == 'Driver is awake and focused' ? 'confidence-good' : 'confidence-bad'}">
                        ${data.camera2.alert}
                      </div>
                    <h3 style="margin: 0;">Gaze Detection</h3>
                    <p style="margin: 0;">
                      <b>Position:</b> ${data.camera2.gaze_center}<br>
                      <b>Status:</b> ${data.camera2.gaze_status}
                    </p>
                    <hr>
                    <h3 style="margin: 0;">Head Movement</h3>
                    <p style="margin: 0;">
                      <b>Pitch:</b> ${data.camera2.pitch}<br>
                      <b>Yaw:</b> ${data.camera2.yaw}<br>
                      <b>Roll:</b> ${data.camera2.roll}<br>
                      <b>Head Status:</b> ${data.camera2.head_status}
                    </p>
                    <hr>
                    <h3 style="margin: 0;">Distraction</h3>
                    <p style="margin: 0;">
                      ${data.camera2.distraction}
                    </p>
                    <hr>
                    <h3 style="margin: 0;">Drowsiness Detection</h3>
                    <p style="margin: 0;">
                      <b>Blinks:</b> ${data.camera2.blinks}<br>
                      <b>Microsleep Duration:</b> ${data.camera2.microsleep_duration}<br>
                      <b>Yawns:</b> ${data.camera2.yawns}<br>
                      <b>Yawn Duration:</b> ${data.camera2.yawn_duration}<br>
                      <b>Blinks Per Minute:</b> ${data.camera2.blinks_per_minute}<br>
                      <b>Yawns Per Minute:</b> ${data.camera2.yawns_per_minute}<br>
                    </p>
                  </div>
                `;
              })
              .catch(err => {
                console.error("Error fetching status:", err);
              });
          }
          // Update status every second
          setInterval(updateStatus, 1000);
          window.onload = updateStatus;
        </script>
      </head>

      <body>
        <div class="container">
          <!-- Camera 1 Card -->
          <div class="card">
            <h2>Camera 1 Feed</h2>
            <div class="camera-container">
              <img src="/video_feed1" alt="Camera 1 Feed"/>
            </div>
            <div id="statusInfo" class="status-panel">
              <!-- status panel 1 updates here -->
            </div>
          </div>

          <!-- Camera 2 Card -->
          <div class="card">
            <h2>Camera 2 Feed</h2>
            <div class="camera-container">
              <img src="/video_feed2" alt="Camera 2 Feed"/>
            </div>
            <div id="statusInfo2" class="status-panel">
              <!-- status panel 2 updates here -->
            </div>
          </div>
        </div>
      </body>
    </html>
    """



# Endpoint for camera 1 stream
@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(camera1, 'cam1'), mimetype='multipart/x-mixed-replace; boundary=frame')

# Endpoint for camera 2 stream
@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames(camera2, 'cam2'), mimetype='multipart/x-mixed-replace; boundary=frame')

#status endpoint to read JSON files and return their contents
@app.route('/status')
def status():
    file1 = "status.json"
    file2 = "status_driver_fatigue.json"
    
    # Read Camera 1 data from file1
    if os.path.exists(file1):
        try:
            with open(file1, "r") as f:
                data1 = json.load(f)
        except Exception as e:
            print("Error reading status.json:", e, flush=True)
            data1 = {
                "per_frame_driver_activity": "Error reading file",
                "per_frame_hands_on_wheel": "N/A",
                "majority_driver_state": "N/A",
                "system_alert": "N/A",
                "hands_monitoring": "N/A",
                "hands_monitoring_confidence": "N/A"
            }
    else:
        data1 = {
            "per_frame_driver_activity": "No data yet",
            "per_frame_hands_on_wheel": "No data yet",
            "majority_driver_state": "No data yet",
            "system_alert": "No data yet",
            "hands_monitoring": "No data yet",
            "hands_monitoring_confidence": "No data yet"
        }
    
    # Read Camera 2 data from file2
    if os.path.exists(file2):
        try:
            with open(file2, "r") as f:
                data2 = json.load(f)
        except Exception as e:
            print("Error reading status_driver_fatigue.json:", e, flush=True)
            data2 = {
                "gaze_center": "Error reading file",
                "gaze_status": "N/A",
                "pitch": "N/A",
                "yaw": "N/A",
                "roll": "N/A",
                "head_status": "N/A",
                "distraction": "N/A",
                "blinks": "N/A",
                "microsleep_duration": "N/A",
                "yawns": "N/A",
                "yawn_duration": "N/A",
                "blinks_per_minute": "N/A",
                "yawns_per_minute": "N/A",
                "alert": "N/A"
            }
    else:
        data2 = {
            "gaze_center": "No data yet",
            "gaze_status": "No data yet",
            "pitch": "No data yet",
            "yaw": "No data yet",
            "roll": "No data yet",
            "head_status": "No data yet",
            "distraction": "No data yet",
            "blinks": "No data yet",
            "microsleep_duration": "No data yet",
            "yawns": "No data yet",
            "yawn_duration": "No data yet",
            "blinks_per_minute": "No data yet",
            "yawns_per_minute": "No data yet",
            "alert": "No data yet"
        }
    
    combined = {
        "camera1": data1,
        "camera2": data2
    }
    return jsonify(combined)


@app.route('/driver_assistant.json')
def driver_assistant():
    filename = 'driver_assistant.json'
    if os.path.exists(filename):
        return send_from_directory(os.getcwd(), filename, mimetype='application/json')
    else:
        return jsonify({"error": "driver_assistant.json not found"}), 404

# ───────────────────────────────
# Start Waitress Production Server
# ───────────────────────────────
if __name__ == '__main__':
    # time.sleep(18)
    serve(app, host='0.0.0.0', port=5000, threads=8) 