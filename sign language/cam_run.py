import cv2
import imutils
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from sklearn.metrics import pairwise
from keras.models import load_model
from skimage.transform import resize
import pyautogui
import time

# Global variables
bg = None

def run_avg(image, accumWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, accumWeight)

def segment(image, threshold=55):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    thresholded = image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def _load_weights():
    try:
        model = load_model("model_sign.h5")
        print(model.summary())
        return model
    except Exception as e:
        return None

def getLetterPrediction(predicted_class):
    letters = "ABCDEFGHIKLMNOPQRSTUVWXY"
    if 0 <= predicted_class < len(letters):
        return letters[predicted_class]
    else:
        return "Unknown Letter"

def getPredictedClass(model):
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = resize(gray_image, (28, 28))
    gray_image = gray_image.reshape(1, 28, 28, 1)
    prediction = model.predict_on_batch(gray_image)
    predicted_class = np.argmax(prediction)
    predicted_letter = getLetterPrediction(predicted_class)
    return predicted_letter

def start_processing():
    global num_frames, calibrated, k

    while True:
        # Get the current frame
        (grabbed, frame) = camera.read()

        # Resize the frame
        frame = imutils.resize(frame, width=700)
        # Flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # Clone the frame
        clone = frame.copy()

        # Get the ROI
        roi = frame[top:bottom, right:left]

        # Convert the ROI to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # To get the background, keep looking until a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successful...")
        else:
            # Segment the hand region
            hand = segment(gray)

            # Check whether the hand region is segmented
            if hand is not None:
                # If yes, unpack the thresholded image and segmented region
                (thresholded, segmented) = hand

                # Draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                # Count the number of fingers
                if k % (fps / 6) == 0:
                    cv2.imwrite('Temp.png', thresholded)
                    predictedClass = getPredictedClass(model)
                    cv2.putText(clone, str(predictedClass), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Show the thresholded image
                cv2.imshow("Thresholded", thresholded)

        k += 1
        # Draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # Increment the number of frames
        num_frames += 1

        # Display the frame with the segmented hand
        cv2.imshow("Video Feed", clone)

        # Write the processed frame to the output video file
        out.write(clone)

        # Observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # If the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

def update_frame():
    _, frame = camera.read()
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    photo = ImageTk.PhotoImage(image=image)
    video_label.imgtk = photo
    video_label.configure(image=photo)
    video_label.after(10, update_frame)

if __name__ == "__main__":
    # Initialize accumulated weight
    accumWeight = 0.5

    # Get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # Get the frame dimensions
    fps = int(camera.get(cv2.CAP_PROP_FPS))
    top, right, bottom, left = 10, 350, 225, 590

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (700, 700))

    # Initialize num of frames
    num_frames = 0
    calibrated = False
    model = _load_weights()
    k = 0

    # Tkinter GUI
    root = tk.Tk()
    root.title("Hand Gesture Recognition")

    # GUI Components
    video_label = ttk.Label(root)
    video_label.pack(padx=10, pady=10)

    start_button = ttk.Button(root, text="Start Processing", command=start_processing)
    start_button.pack(pady=10)

    quit_button = ttk.Button(root, text="Quit", command=root.destroy)
    quit_button.pack(pady=10)

    update_frame()  # Initial call to start updating frames

    root.mainloop()

    # Release resources
    camera.release()
    out.release()
    cv2.destroyAllWindows()
