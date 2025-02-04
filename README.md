# Supermarket Monitoring System

This project focuses on object detection using computer vision to track people in a retail store environment. The system identifies individuals moving within different sections of the store and calculates the time spent in each area.

## Project Workflow

### 1. Upload and Annotate Images
We use **Roboflow** to upload videos and automatically generate annotated images for training. The tool extracts frames at specific intervals and enables annotation for object detection.

### 2. Train the Model
Using the annotated images, we train an object detection model on Roboflow. Pre-processing and augmentation techniques are applied to improve detection accuracy.

### 3. Implement Object Tracking
The script processes video frames to identify people and track their movement between store sections. It:

- Downloads a retail store video from Dropbox.
- Loads the video and extracts frames.
- Sends each frame to the Roboflow API for object detection.
- Draws bounding boxes around detected individuals.
- Determines the section where each person is located.
- Tracks how long each individual stays in a specific area.

### 4. Area Mapping
Lines are drawn on the video frame to separate different store sections. When a person is detected, their position is evaluated relative to these lines to determine their current area.

### 5. Video Output
The processed video is saved with visual annotations, including bounding boxes and section labels. This output helps analyze customer movement within the store.

## Video Snapshots  

### **Before Detection**
The raw video frame before object detection is applied:

![Before Detection](Screenshot%202025-02-04%20110205.png)

### **After Detection**
The processed video frame after object detection and tracking:

![After Detection](Screenshot%202025-02-04%20105401.png)

## Technologies Used

- **OpenCV:** Image and video processing.
- **Roboflow API:** Object detection model deployment.
- **Requests:** API communication and video download.
- **NumPy:** Data processing and calculations.

## Usage

1. Ensure you have Python installed.
2. Install required dependencies using pip.
3. Run the script to process the video and generate results.

## Conclusion
This project demonstrates the power of object detection for tracking people in a retail store. By using Roboflow for model training and OpenCV for processing, we achieve an efficient system for analyzing customer behavior.  
In the future, we can **advance and deploy our model to work with CCTV footage**, allowing real-time monitoring of retail stores.

## Watch the Complete Video  
🔗 **[Click here to watch the full video](https://www.dropbox.com/scl/fi/sttnhv8tnn3p7mjth57yn/processed_downloaded_video.mp4?rlkey=c1igct9ay75h5o3u0ls6l5mtj&st=vrhq7anm&dl=0)**  

