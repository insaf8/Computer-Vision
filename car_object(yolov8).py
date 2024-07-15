import os
import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import pytesseract

# Define class lists
class_list = [0, 1, 2, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
class_list53 = [2, 3, 5, 9, 48, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139]

# Load YOLO models
model = YOLO('yolov8n.pt')
model1 = YOLO('best.pt')
model2 = YOLO('yolov8n-face.pt')

# Define fixed colors for each model
model_colors = {
    model: (255, 0, 0),      # Red for model
    model1: (0, 0, 0),     
    model2: (0, 0, 255),     # Blue for model2
}

# Initialize tkinter window
window = tk.Tk()
window.title("Object Detection")

# Create a canvas to display processed images or videos
canvas = tk.Canvas(window, width=640, height=640)
canvas.pack()

def preprocess_image_for_ocr(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.GaussianBlur(binary, (7, 7), 0)

    # Dilate the binary image to improve text detection
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=4)

    # Denoise image using Gaussian blur
    
    return dilated


# Function to perform OCR on the detected billboard image
def extract_text_from_image(image, save_path, index):
    preprocessed_image = preprocess_image_for_ocr(image)
    text = pytesseract.image_to_string(preprocessed_image, config='--psm 6')
    text_file_path = os.path.join(save_path, f"billboard_{index}.txt")
    with open(text_file_path, 'w') as file:
        file.write(text)
# Function to process and save images
def process_image():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename()
    
    if file_path:
        # Read image using OpenCV
        image = cv2.imread(file_path)
        
        # Predict using the first and second YOLO models
        results = model.predict(image)
        results1 = model1.predict(image)
        results2 = model2.predict(image)
        
        # Initialize Annotator with the original image
        annotator = Annotator(image)
        
        # Create folders to save detected faces and billboards
        save_folder_faces = os.path.join(os.path.dirname(file_path), "detected_faces")
        os.makedirs(save_folder_faces, exist_ok=True)
        save_folder_billboards = os.path.join(os.path.dirname(file_path), "detected_billboards")
        os.makedirs(save_folder_billboards, exist_ok=True)
        
        # Save detected faces
        for result in results2:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                face_image = image[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                cv2.imwrite(os.path.join(save_folder_faces, f"face_{i}.jpg"), face_image)
        
        # Save detected billboards and annotate detections from the models
        for r, mdl in zip([results, results1, results2], [model, model1, model2]):
            for result in r:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    c = box.cls
                    color = model_colors[mdl]

                    annotator.box_label(b, mdl.names[int(c)], color=color)
                    
                    if mdl == model1 and int(c) == 1:  # Check if the model is model1 and the class label is 1 for billboards
                        billboard_image = image[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                        cv2.imwrite(os.path.join(save_folder_billboards, f"billboard_{i}.jpg"), billboard_image)
                        extract_text_from_image(billboard_image, save_folder_billboards, i)
        
        # Get the annotated image
        annotated_img = annotator.result()
        
        # Convert the annotated image to RGB format for display
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Convert image from OpenCV BGR format to PIL format
        image_pil = Image.fromarray(annotated_img_rgb)
        
        # Resize image to fit canvas
        max_width = 640
        max_height = 640
        if image_pil.width > max_width or image_pil.height > max_height:
            image_pil.thumbnail((max_width, max_height))
        
        # Convert PIL image to ImageTk format for tkinter
        photo_image = ImageTk.PhotoImage(image_pil)
        
        # Clear previous content from canvas
        canvas.delete("all")
        
        # Display image on canvas
        canvas.create_image(0, 0, anchor=tk.NW, image=photo_image)
        canvas.image = photo_image  # Keep a reference to prevent garbage collection

# Function to process and save video frames
def process_video():
    # Open file dialog to select a video file
    file_path = filedialog.askopenfilename()
    
    if file_path:
        # Initialize the video capture
        cap = cv2.VideoCapture(file_path)
        
        # Create folders to save detected faces and billboards
        save_folder_faces = os.path.join(os.path.dirname(file_path), "detected_faces")
        os.makedirs(save_folder_faces, exist_ok=True)
        save_folder_billboards = os.path.join(os.path.dirname(file_path), "detected_billboards")
        os.makedirs(save_folder_billboards, exist_ok=True)
        
        # Define function to process and display frames
        def process_and_display_frame():
            ret, frame = cap.read()
            if ret:
                # Predict using the first and second YOLO models
                results = model.predict(frame)
                results1 = model1.predict(frame)
                results2 = model2.predict(frame)
                
                # Initialize Annotator with the original frame
                annotator = Annotator(frame)
                
                # Save detected faces
                for result in results2:
                    boxes = result.boxes
                    for i, box in enumerate(boxes):
                        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                        face_image = frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                        cv2.imwrite(os.path.join(save_folder_faces, f"face_{i}.jpg"), face_image)
                
                # Save detected billboards and annotate detections from the models
                for r, mdl in zip([results, results1, results2], [model, model1, model2]):
                    for result in r:
                        boxes = result.boxes
                        for i, box in enumerate(boxes):
                            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                            c = box.cls
                            color = model_colors[mdl]

                            annotator.box_label(b, mdl.names[int(c)], color=color)
                            
                            if mdl == model1 and int(c) == 1:  # Check if the model is model1 and the class label is 1 for billboards
                                billboard_image = frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                                cv2.imwrite(os.path.join(save_folder_billboards, f"billboard_{i}.jpg"), billboard_image)
                                extract_text_from_image(billboard_image, save_folder_billboards, i)
                
                # Convert the frame to HSV color space
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Define range for green color in HSV
                lower_green = np.array([30, 100, 100])
                upper_green = np.array([85, 255, 255])
                
                # Create a mask for green areas
                mask = cv2.inRange(hsv_frame, lower_green, upper_green)
                
                # Use dilation to merge nearby green areas
                kernel = np.ones((7, 7), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=10)
                
                # Find contours of the green areas
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # Annotate merged green areas as trees
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Tree", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Get the annotated frame
                annotated_frame = annotator.result()
                
                # Convert the frame to the format expected by YOLOv5 (BGR to RGB)
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Convert the NumPy array to a PIL Image
                frame_pil = Image.fromarray(frame_rgb)
                
                # Resize image to fit canvas
                max_width = 640
                max_height = 640
                if frame_pil.width > max_width or frame_pil.height > max_height:
                    frame_pil.thumbnail((max_width, max_height))
                
                # Convert PIL image to ImageTk format for tkinter
                photo_image = ImageTk.PhotoImage(frame_pil)
                
                # Display frame on canvas
                canvas.create_image(0, 0, anchor=tk.NW, image=photo_image)
                canvas.image = photo_image  # Keep a reference to prevent garbage collection
                
                # Schedule processing and displaying of the next frame
                window.after(1, process_and_display_frame)
            else:
                # Release the video capture and close all OpenCV windows
                cap.release()
                cv2.destroyAllWindows()
        
        # Start processing and displaying frames
        process_and_display_frame()

# Add buttons to upload image or video
image_button = tk.Button(window, text="Upload Image", command=process_image)
image_button.pack(pady=10)

video_button = tk.Button(window, text="Upload Video", command=process_video)
video_button.pack(pady=10)

# Run the tkinter event loop
window.mainloop()
