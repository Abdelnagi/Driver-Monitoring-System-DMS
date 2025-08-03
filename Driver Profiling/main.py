import time
from Recognizer import Config, FaceDatabase, FaceRecognizer, label_image_write
import os
import cv2
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 1. Initialize configuration and classes
config = Config()
face_db = FaceDatabase(config)
recognizer = FaceRecognizer(config, face_db)

Data_path ="./Images/Data"
Valid_Test_path ="./Images/Test"
query_image_path = "./Images/Test/Ahmad Test.jpg"

# Scales the image down with aspect ratio preserved for SAVING iamges
def resize_keep_aspect(image, max_size=600):
    h, w = image.shape[:2]
    scale = min(max_size / w, max_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def image_show(query_img, name, real_name, accuracy):
    # Display the result
    query_img = resize_keep_aspect(query_img)
    result_img = label_image_write(query_img, f"{name} (Accuracy: {accuracy:.2f})")
    img_path = os.path.join(Data_path, f"{real_name}.jpg")
    print(f"Saved Result at {img_path}")
    cv2.imwrite(img_path, result_img)
    #cv2.imshow(f"Recognition Result of {real_name}", result_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def image_search(Valid_Test_path):
    if os.path.exists(Valid_Test_path):
        real_name =  os.path.splitext(os.path.basename(Valid_Test_path))[0]
        query_img = cv2.imread(Valid_Test_path)
        name, distance = recognizer.find_closest_match(query_img)
        image_show(query_img, name, real_name, distance)
    else:
        print(f"Query image '{Valid_Test_path}' not found. Skipping match test.")

def folder_search():
    for image in os.listdir(Valid_Test_path):
        query_image_path = os.path.join(Valid_Test_path, image)
        image_search(query_image_path)

# App loop
start = time.time()
app =("Register Folder", "Register Frame", "Search Folder", "Search Image")

session = app[2]

if session == app[0]: recognizer.register_faces_from_folder(Data_path)
elif session == app[1]: recognizer.get_camera_frame(0,"Nagi", False)
elif session == app[2]: folder_search()
elif session == app[3]: image_search(query_image_path)

end = time.time()
print(f"Finished in: {end - start:.4f} seconds")

