import multiprocessing
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import sys
import time

# Ensure multiprocessing compatibility
multiprocessing.set_start_method("spawn", force=True)

# Define ResNet-18 Model for Activity Detection
class CustomResNet18(nn.Module):
    def __init__(self, model_path, labels, classes, device="CPU"):
        super(CustomResNet18, self).__init__()
        self.device = device
        self.model = self.load_model(model_path, len(classes))
        self.model.eval()
        self.labels = labels
        self.classes = classes
        self.model = self.model.to(self.device)

    def load_model(self, model_path, num_classes):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        return model

# Global Variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"D:\downloads\model_ft.pth"
labels = list(range(0, 10))
class_labels  = {
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

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def capture_frames(frame_queue, stop_event, video_path):
    """Capture video frames and store them in the queue while maintaining real-time speed."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps if fps > 0 else 1 / 30  # Default to 30 FPS if unknown

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            if frame_queue.qsize() < 2:
                frame_queue.put(frame)
            time.sleep(frame_time)
        else:
            print("End of video. Exiting.")
            stop_event.set()
            break
    cap.release()

def process_frames(frame_queue, result_queue, stop_event):
    """Process frames for real-time activity detection."""
    model = CustomResNet18(model_path, labels, class_labels, device=device).model
    #print("Processing started...")  # Debugging print
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            #print("Processing a frame...")  # Debugging print
            top3_predictions = predict_activity(frame, model)
            result_queue.put((frame, top3_predictions))
        except Exception as e:
            print(f"Error in activity prediction: {e}")

def predict_activity(frame, model):
    """Predict driver activity using ResNet-18."""
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probabilities = F.softmax(outputs, dim=1)
            top3_probs, top3_indices = torch.topk(probabilities, 3)

            top3_labels = [class_labels[idx.item()] for idx in top3_indices[0]]
            top3_confidences = [round(prob.item() * 100, 2) for prob in top3_probs[0]]

            return list(zip(top3_labels, top3_confidences))

    except Exception as e:
        print(f"Error in activity prediction: {e}")
        return "Unknown"

class ActivityDetection(QMainWindow):
    def __init__(self, frame_queue, result_queue, stop_event, video_path):
        super().__init__()

        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event

        self.alert_text = ""
        self.driver_state = ""

        self.setWindowTitle("Driver Activity Detection")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: white;")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("border: 2px solid black;")
        self.video_label.setFixedSize(640, 480)
        self.layout.addWidget(self.video_label)

        self.info_label = QLabel()
        self.info_label.setStyleSheet("background-color: white; border: 1px solid black; padding: 10px;")
        self.layout.addWidget(self.info_label)

        self.update_info()

        self.timer = self.startTimer(30)  # 30ms update interval

    def update_info(self):
        """Update activity info on UI."""
        info_text = (
            f"<div style='font-family: Arial, sans-serif; color: #333;'>"
            f"<h2 style='text-align: center; color: #4CAF50;'>Driver Activity Detection</h2>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"{self.alert_text}"
            f"<p><b> Current Driver State:</b> {self.driver_state}</p>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"</div>"
        )
        self.info_label.setText(info_text)

    def timerEvent(self, event):
        """Update UI with processed frame results."""
        try:
            frame, predictions = self.result_queue.get_nowait()

            #print(f"Updating UI with: {predictions}")  # Debugging print

            if predictions:
                self.driver_state = predictions[0][0]  # Set the highest-confidence prediction
                self.alert_text = f"<b>Alert:</b> {self.driver_state} detected!" if self.driver_state != "Safe driving" else ""

                confidence_display = "<br>".join([f"{cls}: {conf}%" for cls, conf in predictions])

                self.info_label.setText(
                    f"<b>Current Driver State:</b> {self.driver_state}<br>"
                    f"<b>Top Predictions:</b><br>{confidence_display}"
                )

                self.display_frame(frame)  # Ensure the frame is updated

        except Exception as e:
            print(f"UI Update Error: {e}")


    def display_frame(self, frame):
        """Display frame in PyQt5 GUI."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    start_time = time.time()
    app = QApplication(sys.argv)

    video_path, _ = QFileDialog.getOpenFileName(None, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
    if not video_path:
        print("No video selected. Exiting.")
        sys.exit()

    frame_queue = multiprocessing.Queue(maxsize=2)
    result_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    capture_process = multiprocessing.Process(target=capture_frames, args=(frame_queue, stop_event, video_path))
    process_process = multiprocessing.Process(target=process_frames, args=(frame_queue, result_queue, stop_event))

    capture_process.start()
    process_process.start()

    window = ActivityDetection(frame_queue, result_queue, stop_event, video_path)
    window.show()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    sys.exit(app.exec_())
