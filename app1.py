import os
import cv2
import requests
from PIL import Image
import numpy as np
import io
from requests_toolbelt.multipart.encoder import MultipartEncoder
from inference_sdk import InferenceHTTPClient  # Import your inference SDK

# Create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=""  # Replace with your actual API key
)

# Paths and settings
TO_PREDICT_PATH = "Images/Prediction_Images/To_Predict/"
PREDICTED_PATH = "Images/Prediction_Images/Predicted_Images/"
MIN_SCORE = 0.5
LOCAL_VIDEO_PATH = "downloaded_video.mp4"  # Local path to save the video
PREDICTED_VIDEO_PATH = "predictions/"  # Directory for predicted video frames
DROPBOX_VIDEO_URL = "https://www.dropbox.com/scl/fi/r2kw1ptclywmzvotva86t/Retail_Store.mp4?rlkey=tzg1x8ydukwp7aaycichmti56&st=2g0xgzj2&dl=1"  # Update with your link

# Helper function to draw a line on an image
def draw_line(image, xf1, yf1, xf2, yf2):
    h, w = image.shape[:2]  # Get image dimensions
    start_point = (int(w * xf1), int(h * yf1))
    end_point = (int(w * xf2), int(h * yf2))

    # Calculate the slope (m) and intercept (b) of the line equation: y = mx + b
    if xf2 - xf1 != 0:
        slope = (yf2 - yf1) / (xf2 - xf1)
        b = yf1 - slope * xf1
        print(f"Line equation: y = {round(slope, 3)}x + {round(b, 3)}")
    else:
        print("Vertical line")

    cv2.line(image, start_point, end_point, (255, 0, 0), 4)


# Helper function to write the area name on the image
def writes_area_text(image, text, xf1, yf1):
    w = image.shape[1]
    h = image.shape[0]
    start_point = (int(w * xf1), int(h * yf1))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    color = (255, 100, 100)
    thickness = 2

    # Draw background text
    cv2.putText(image, text, start_point, font, fontScale, (0, 0, 0), thickness + 3)
    # Draw foreground text
    cv2.putText(image, text, start_point, font, fontScale, color, thickness)

# Function to download video from Dropbox
def download_video_from_dropbox(url, local_path):
    print(f"Downloading video from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"Video downloaded successfully: {local_path}")
        return True
    print(f"Failed to download video. Status code: {response.status_code}")
    return False

# Function to infer objects in the image using the Roboflow model
def get_predictions(image):
    # Save the image temporarily
    cv2.imwrite("temp_image.jpg", image)
    prediction = CLIENT.infer(
        "temp_image.jpg",
        model_id="supermarket_detection/2"  # Replace with your model ID
    )
    return prediction

# Function to determine which area an object is located in

def which_area(image, midx, midy):
    w = image.shape[1]
    h = image.shape[0]
    xf = midx / w
    yf = midy / h

    x1, x2, x3, x4, x5, x6 = 0.10, 0.30, 0.35, 0.55, 0.65, 0.85
    y1 = 0.0 * xf + 0.2  # Top-left line
    y2 = -0.444 * xf + 0.294  # Top-middle line
    y3 = 2.75 * xf + -0.025  # Left line
    y4 = -1.0 * xf + 1.1  # Bottom line
    y5 = 1.0 * xf + -0.2  # Middle Line

    if xf <= x1:
        area = "A2" if yf <= y1 else "Register"
    elif xf > x1 and xf <= x2:
        area = "A2" if yf <= y2 else "A3" if yf <= y3 else "Register"
    elif xf > x2 and xf <= x3:
        area = "A2" if yf <= y2 else "Area 3" if yf <= y4 else "Entrance"
    elif xf > x3 and xf <= x4:
        area = "A2" if yf <= y2 else "A1" if yf <= y5 else "A3" if yf <= y4 else "Entrance"
    elif xf > x4 and xf <= x5:
        area = "A1" if yf <= y5 else "A3" if yf <= y4 else "Entrance"
    elif xf > x5 and xf <= x6:
        area = "A1" if yf <= y4 else "Entrance"
    else:
        area = "Entrance"

    return area


# Video processing function
def process_video():
    if not download_video_from_dropbox(DROPBOX_VIDEO_URL, LOCAL_VIDEO_PATH):
        return

    video_capture = cv2.VideoCapture(LOCAL_VIDEO_PATH)
    if not video_capture.isOpened():
        print(f"Failed to open video {LOCAL_VIDEO_PATH}")
        return

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    success, frame = video_capture.read()

    if not success:
        print(f"Error reading the first frame.")
        return

    if not os.path.exists(PREDICTED_VIDEO_PATH):
        os.makedirs(PREDICTED_VIDEO_PATH)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(
        os.path.join(PREDICTED_VIDEO_PATH, 'processed_' + os.path.basename(LOCAL_VIDEO_PATH)),
        fourcc, video_fps, (frame.shape[1], frame.shape[0])
    )

    frame_idx = 1
    while success:
        print(f"Processing frame {frame_idx} of {frame_count}...")

        # Draw area boundary lines on each frame
        draw_line(frame, 0.00, 0.20, 0.10, 0.20)  # Top-left line
        draw_line(frame, 0.10, 0.25, 0.55, 0.05)  # Top-middle line
        draw_line(frame, 0.10, 0.25, 0.30, 0.80)  # Left line
        draw_line(frame, 0.35, 0.15, 0.65, 0.45)  # Middle Line
        draw_line(frame, 0.30, 0.80, 0.85, 0.25)  # Bottom line
        draw_line(frame, 0.55, 0.05, 0.85, 0.25)  # Right line

        # Get predictions for the frame
        pred = get_predictions(frame)

        # Process the predictions and draw bounding boxes
        if 'predictions' in pred and pred['predictions']:
            for p in pred['predictions']:
                x1, y1, x2, y2 = p['x'], p['y'], p['width'], p['height']
                area = which_area(frame, x1, y1)
                print(f"Object detected at ({x1}, {y1}). Area: {area}")

                cv2.rectangle(frame, (int(x1 - x2 / 2), int(y1 - y2 / 2)),
                              (int(x1 + x2 / 2), int(y1 + y2 / 2)), (0, 255, 0), 2)
                
                cv2.putText(frame, area, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 0, 0), 2)

        # Write processed frame to output video
        video_out.write(frame)

        success, frame = video_capture.read()
        frame_idx += 1

    video_capture.release()
    video_out.release()
    print("Video processing completed and saved.")


# Start processing the video
process_video()
