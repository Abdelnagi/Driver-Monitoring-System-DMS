import os
import cv2
import pandas as pd
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import time
# --- Configuration Constants ---
# Using a class for configuration makes it cleaner to pass around
class Config:
    DB_PATH = "./Database/face_database.feather"
    FACE_IMG_DIR = "./Database/stored-faces"
    DEMO_RESULTS_DIR = "./Database/Demo Results"
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    MAX_IMAGES_PER_USER = 3
    DUPLICATE_THRESHOLD = 1e-3  # Cosine distance threshold; lower is more similar
    RECOGNITION_THRESHLD = 0.6
    EMBEDDING_DIMENSION = 512
# --- Standardized Logging ---
def log(msg, is_good=True):
    state = "GOOD" if is_good else "BAD"
    print(f"[INFO] [{state}] {msg}")

# --- Utility Functions ---
def label_image_write(image, text, position=(20, 50), font=cv2.FONT_HERSHEY_SIMPLEX,
                      font_scale=0.6, text_color=(0, 0, 255),
                      bg_color=(0, 0, 0), bg_opacity=0.7, thickness=1, padding=5):
    # (This function remains unchanged as it's a self-contained utility)
    overlay = image.copy()
    output = image.copy()
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x, y = position
    rect_tl = (x - padding, y - text_h - padding)
    rect_br = (x + text_w + padding, y + padding)
    cv2.rectangle(overlay, rect_tl, rect_br, bg_color, cv2.FILLED)
    cv2.addWeighted(overlay, bg_opacity, output, 1 - bg_opacity, 0, output)
    cv2.putText(output, text, (x, y), font, font_scale, text_color, thickness)
    return output

# ==============================================================================
#                       DATABASE MANAGEMENT
# Handles all reading and writing of face data using Feather format.
# ==============================================================================
class FaceDatabase:
    def __init__(self, config):
        self.config = config
        self.db_path = config.DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(config.FACE_IMG_DIR, exist_ok=True)
        self.df = self._load_database()

    def _load_database(self):
        """Loads the face database from a Feather file."""
        if os.path.exists(self.db_path):
            log(f"Loading database from {self.db_path}")
            return pd.read_feather(self.db_path)
        else:
            log("No existing database found. Creating a new one.")
            return pd.DataFrame(columns=["username", "embedding", "path"])

    def _save_database(self):
        """Saves the entire DataFrame to the Feather file, overwriting it."""
        self.df.to_feather(self.db_path)
        log("Database saved successfully.")

    def is_unique(self, username, query_embedding):
        """Checks if a new face is unique enough to be added for a user."""
        if self.df.empty:
            return True , 0

        # Filter for embeddings of the same user
        user_faces = self.df[self.df["username"] == username]
        user_entries = len(user_faces)
        # 1. Check if user has reached the maximum image count
        if  user_entries >= self.config.MAX_IMAGES_PER_USER:
            log(f"{username}'s Quota exceeded. User already has {self.config.MAX_IMAGES_PER_USER} images.", is_good=False)
            return False , user_entries 

        # 2. Check for duplicate images
        is_duplicate_found = (user_faces["embedding"]
                            .apply(lambda stored_embedding: cosine(query_embedding, stored_embedding)) 
                            < self.config.DUPLICATE_THRESHOLD).any()

        if is_duplicate_found:
            log(f"Duplicate image detected for {username}. Please use a new image", is_good=False)
            return False, user_entries
        return True, user_entries
    
    def save_iamge(self, username, user_count, face_image_rgb):
                # Save the cropped face image
        image_count = user_count + 1
        img_filename = f"{username}_{image_count}.jpg"
        img_path = os.path.join(self.config.FACE_IMG_DIR, img_filename)
        cv2.imwrite(img_path, face_image_rgb)
        log(f"Saved {username}'s face image to: {img_path}")
        return img_path

    def add_face(self, username, embedding, face_image_rgb, isFolder=False):
        """
        Adds a new face to the database. This method handles 
        uniqueness checks, saving the image, and updating the database.
        """
        uniqness, user_count = self.is_unique(username, embedding)
        if not uniqness: return False

        img_path = self.save_iamge(username, user_count, face_image_rgb)
        # Add new entry to the DataFrame
        new_entry = pd.DataFrame([{"username": username, 
                                   "embedding": embedding,
                                   "path": img_path}])
        self.df = pd.concat([self.df, new_entry], ignore_index=True)

        # Save the updated database
        self._save_database() 
        
        return True

# ==============================================================================
# CLASS 2: FACE RECOGNITION LOGIC
# Handles all image processing, face detection, and matching.
# ==============================================================================
class FaceRecognizer:
    def __init__(self, config, face_database):
        self.config = config
        self.db = face_database
        log("Initializing Face Analysis model...")
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0)  # -1 for CPU, 0 for GPU
        log("Face Analysis model initialized.")

    def get_face_info(self, image_rgb):
        """Detects a single face and returns its cropped image and embedding."""
        faces = self.app.get(image_rgb)
        
        if len(faces) == 0:
            log("No face detected.", is_good=False)
            #return None, None
        if len(faces) > 1:
            log(f"Multiple faces ({len(faces)}) detected. Selecting the most dominant (largest) face.")
            
            dominant_face = None
            max_area = 0
            
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                
                if area > max_area:
                    max_area = area
                    dominant_face = face
            
            # After the loop, dominant_face will hold the face with the largest bbox
            face = dominant_face
        else:
            # If there's only one face, just use it
            face = faces[0]

        #face = faces[0]
        embedding = face.embedding
        bbox = face.bbox.astype(int)
        # Crop from the original RGB image
        cropped_face = image_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return cropped_face, embedding

    def register_face_path(self, image_path, isFolder=False):
        """Processes and registers a single image file."""
        file_name = os.path.basename(image_path)
        if os.path.splitext(file_name)[1].lower() not in self.config.VALID_EXTENSIONS:
            log(f"Failed to register {file_name}. File extention not valid.", is_good=False)
            return
                
        log(f"Registering face from: {image_path}")
        username = os.path.splitext(file_name)[0]
        rbg_image = cv2.imread(image_path)
        if rbg_image is None:
            log(f"Could not read image: {image_path}", is_good=False)
            return

        cropped_face, embedding = self.get_face_info(rbg_image)
        if embedding is not None:
            if isFolder: return username, embedding, cropped_face
            elif self.db.add_face(username, embedding, cropped_face):
                log(f"Successfully registered {username}.")
            else:
                log(f"Failed to register {username}.", is_good=False)
                return

    def register_face_image(self, rbg_image, username):
        """Processes and registers a single image file."""
        log(f"Registering face for {username} from Frame.")
        if rbg_image is None:
            log(f"Could not read image", is_good=False)
            return

        cropped_face, embedding = self.get_face_info(rbg_image)
        if embedding is not None:
            if self.db.add_face(username, embedding, cropped_face):
                log(f"Successfully registered {username}.")
            else:
                log(f"Failed to register {username}.", is_good=False)

    def register_faces_from_folder(self, folder_path):
        """Processes all valid images in a given folder."""
        log(f"Starting to process folder: {folder_path}")
        new_faces_added = 0

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            username, embedding, cropped_face = self.register_face_path(file_path, True)
            
            uniqness, user_count = self.db.is_unique(username, embedding)
            if not uniqness: return False

            img_path = self.db.save_iamge(username, user_count, cropped_face)
            # Add new entry to the DataFrame
            new_entry = pd.DataFrame([{"username": username, 
                                   "embedding": embedding,
                                   "path": img_path}])
            self.db.df = pd.concat([self.db.df, new_entry], ignore_index=True)

            log(f"Queued '{filename}' for addition to database.")
            new_faces_added += 1

        # ---  Save Operation ---
        # After the loop, if we have new faces, save the entire updated DataFrame once.
        if new_faces_added > 0:
            log(f"Batch processing complete. Added {new_faces_added} new faces.")
            self.db._save_database() # The single save operation!
        else:
            log("Batch processing complete.")
       

    def get_camera_frame(self, camera_index=0, name=None, save=False):
        """Perform real-time face recognition using webcam feed."""
        # Initialize video capture
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Cannot open camera")
            return
        while True:
            register = False
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                log("Can't receive frame. Exiting...", False)
                break
            
            if save:
                frame = label_image_write(frame, "Press S to save Face")
                cv2.imshow('Face Registration', frame)
                if cv2.waitKey(1) == ord('s'):    
                    self.register_face_image(frame, name)
                elif cv2.waitKey(1) == ord('v'):
                    save = False
                    continue
            else:    
                start = time.time()
                name, distance = self.find_closest_match(frame)
                if name is None or distance is None:
                    log("Program Failed: Insufficient Input")
                    return None
                label = f"{name} (Distance: {distance:.2f})"
                frame = label_image_write(frame, label)
                end =time.time()
                log(f"---- Inference Time = {(end - start):.2f} seconds")
                cv2.imshow('Face Recognition', frame)
            
                # Break the loop on 'q' key
                if cv2.waitKey(1) == ord('q'):
                    break

        
    




    def find_closest_match(self, query_image_rgb):
        """Finds the closest matching face in the database for a query image."""
        if self.db.df.empty:
            log("Cannot find match, the face database is empty.", is_good=False)
            return None, None

        _, query_embedding = self.get_face_info(query_image_rgb)
        if query_embedding is None:
            log("Could not get embedding for the query image.", is_good=False)
            return None, None

        # --- Vectorized Search Logic ---
        # 1. Calculate cosine distance from the query_embedding to ALL stored embeddings at once.
        #    This returns a new pandas Series containing all the distances.
        distances = self.db.df["embedding"].apply(
            lambda stored_embedding: cosine(query_embedding, stored_embedding)
            )

        # 2. Find the minimum distance value in the entire Series and its index and Apply the recognition threshold.
        min_distance = distances.min()
        best_match_idx = distances.idxmin()

        if min_distance > self.config.RECOGNITION_THRESHLD:
            log(f"No confident match found. Closest distance ({min_distance:.4f}) is above threshold ({self.config.RECOGNITION_THRESHLD}).", is_good=False)
            return "Unknown", min_distance

        # 3. Use the index to look up the corresponding username in the original DataFrame.
        best_match_name = self.db.df.loc[best_match_idx, "username"]
        # --- End of Vectorized Logic ---
        #accuracy = (1 + abs(min_distance)) * 100 / 2       \t(Accuracy = {accuracy:.4f}%)
        log(f"Closest match: {best_match_name}  \t(Distance = {min_distance:.4f})")
        return best_match_name, min_distance

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # 1. Initialize configuration and classes
    config = Config()
    face_db = FaceDatabase(config)
    recognizer = FaceRecognizer(config, face_db)

    Data_path ="./Images/Data/2.jpg"
    Valid_Test_path ="./Images/Test/inData"
    inValid_Test_path ="./Images/Test/not inData"
    query_image_path = "./Images/Test/inData"

    # 2. Example Usage: Register all faces from an input folder
    # Create a folder named "input_images" and place your jpg/png files there.
    # Name the files like "JohnDoe.jpg", "JaneSmith-1.png", etc.

    #recognizer.register_faces_from_folder(Data_path)

    # 3. Example Usage: Find a match for a new query image
    # Place an image you want to test in the main directory as "query.jpg"
    def image_search(query_image_path):
        if os.path.exists(query_image_path):
            real_name =  os.path.splitext(os.path.basename(query_image_path))[1].lower()
            query_img = cv2.imread(query_image_path)
            if query_img is not None:
                name, accuracy ,distance = recognizer.find_closest_match(query_img)
                if name:
                    # Display the result
                    result_img = label_image_write(query_img, f"{name} (Accuracy: {accuracy:.2f})")
                    cv2.imshow(f"Recognition Result of {real_name}", result_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        else:
            print(f"Query image '{query_image_path}' not found. Skipping match test.")

    # 4. Example Usage: Find a match for a every new query image inside a folder

    def folder_search():
        for image in os.listdir(Valid_Test_path):
            query_image_path = os.path.join(Valid_Test_path, image)
            image_search(query_image_path)

    def image_show(query_img, name, real_name, accuracy):
        # Display the result
        result_img = label_image_write(query_img, f"{name} (Distance: {accuracy:.2f})")
        img_path = os.path.join(Data_path, f"{real_name}.jpg")
        print(f"Saved Result at {img_path}")
        cv2.imwrite(img_path, result_img)
        cv2.imshow(f"Recognition Result of {real_name}", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    img = r"E:\My CUFE\Graduation Project\Face Regocnition New\The Data\Nagi.png"
    img = cv2.imread(img)
    name, distance = recognizer.find_closest_match(img)

    image_show(img, name, "Nagi", distance)