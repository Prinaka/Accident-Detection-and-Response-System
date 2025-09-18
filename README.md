# Accident-Detection-and-Response-System
This project provides an automated solution for detecting potential vehicle collisions in video feeds. It uses object detection and tracking to identify vehicles and monitor their movements. When a collision is detected, the system logs the event, saves a snapshot of the moment, and alerts the user.

**Key Features:**
* Real-time Object Detection: Utilises the pre-trained YOLOv8m model to identify various types of vehicles (cars, trucks, buses, etc.).
* Vehicle Tracking: Assigns unique IDs to tracked vehicles, allowing for the calculation of their relative movements and speeds.
* Collision Detection: Uses Intersection over Union (IoU) to detect when the bounding boxes of two moving vehicles overlap significantly.
* Velocity-Based Filtering: Reduces false positives by only flagging collisions between vehicles that are in motion.
* Multi-Modal Alerts: Notifies the user of a detected collision with a pop-up alert and an audible beep.
* Data Logging: Automatically logs collision events, including timestamps, vehicle IDs, and bounding box coordinates, to a CSV file (collision_log.csv) and a text file (collision_log.txt).
* Evidence Collection: Saves a frame of each collision event as a JPG file in the collision_frames directory for later review.
* Web Dashboard: Provides a user-friendly dashboard built with Streamlit to view recent collisions, examine saved frames, and see basic analytics like collisions per hour.

**Installation:**

1. Clone the repository:
```
gh repo clone Prinaka/Accident-Detection-and-Response-System
cd Accident-Detection-and-Response-System
```

2. Create and activate a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Prepare Data Folders:
* Ensure your video file is named video3.mp4 and is located in the root directory. If your video has a different name, update the video_path variable in main.py.

**Usage:**

* Run the Accident Detection script:
```
python main.py
```
* Run the Streamlit app:
```
streamlit run app.py
```

**File Overview:**

* collision_frames/ : Directory to store saved collision images
* collision_log.csv : CSV file to log collision data
* collision_log.txt : Text file to log collision messages
* video3.mp4 : Sample video for testing (user provided)
* main.py : The core Python script for detection and logging
* app.py : The Streamlit application for the dashboard

**Customization & Extending:**

* The winsound library is specific to Windows. If you are running this code on macOS or Linux, you will need to replace the winsound.Beep() function with an alternative for sound alerts.
* This system can be easily adapted for real-time monitoring. Instead of reading from a video file, you can modify the video_path variable in main.py to point to a live surveillance camera stream.
* The performance of the system is highly dependent on the quality and resolution of the input video and the complexity of the scene.
* The iou_threshold can be adjusted to fine-tune the sensitivity of the collision detection. A lower value will detect near-misses, while a higher value will require a more significant overlap.
* The cooldown_time variable prevents repeated notifications for the same ongoing collision. You can modify this value (in seconds) to suit your needs.
  
**License:**

This project is licensed under the MIT License – see the LICENSE file for details.
