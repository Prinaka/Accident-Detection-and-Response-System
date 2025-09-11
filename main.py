!pip install opencv-python
!pip install ultralytics

from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import ultralytics
from ultralytics import YOLO
import numpy as np
from google.colab.patches import cv2_imshow
import time
import smtplib
from email.mime.text import MIMEText

model = YOLO('yolov8m.pt')

video_path = '/content/drive/MyDrive/Colab Notebooks/videos/your_video.mp4'  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)

velocity_threshold = 1.5  

previous_positions = {}


def get_object_velocity(object_id, x1, y1, x2, y2):
    current_center = np.array([
        (x1.cpu().numpy() + x2.cpu().numpy()) / 2,  
        (y1.cpu().numpy() + y2.cpu().numpy()) / 2   
    ])

    if object_id in previous_positions:
        previous_center = previous_positions[object_id]
        distance = np.linalg.norm(current_center - previous_center)
        velocity = distance
    else:
        velocity = 0

    previous_positions[object_id] = current_center

    return velocity


def log_collision(vehicle1_id, vehicle2_id, bbox1, bbox2):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_message = (f"Collision detected between Vehicle {vehicle1_id} and Vehicle {vehicle2_id} at {current_time}.\n"
                   f"Vehicle 1 bounding box: {bbox1}\n"
                   f"Vehicle 2 bounding box: {bbox2}\n\n")

    with open('collision_log.txt', 'a') as log_file:
        log_file.write(log_message)

    print("Collision logged:", log_message)

email_counter = 0
MAX_EMAILS = 3  

def send_email_notification(vehicle1_id, vehicle2_id, bbox1, bbox2):
    global email_counter

    if email_counter >= MAX_EMAILS:
        return

    sender_email = "your_gmailid@gmail.com"   # Replace with email id
    receiver_email = "receiver_gmailid@gmail.com"   # Replace with receiver's email id
    subject = "COLLISION ALERT!!"

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    message_content = (f"A collision has been detected between Vehicle {vehicle1_id} and Vehicle {vehicle2_id} at {current_time}.\n")

    msg = MIMEText(message_content)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    smtp_server = "smtp.gmail.com" 
    smtp_port = 587
    smtp_user = "your_gmailid@gmail.com"   # Replace with email id
    smtp_password = "your_app_password"   # Replace with your app password

    try:
        # Establish connection to the SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.set_debuglevel(1)  
        server.starttls() 
        server.login(smtp_user, smtp_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())

        print(f"Email sent to {receiver_email} about the collision.")
        email_counter += 1
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        if 'server' in locals() and server:
            server.quit()


def check_collision(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Check if bounding boxes overlap
    if (x1_1 < x2_2 and x2_1 > x1_2) and (y1_1 < y2_2 and y2_1 > y1_2):
        return True
    return False



# Process the video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break  

    # Perform object detection using YOLOv8
    results = model(frame)

    vehicles = []
    for detection in results[0].boxes:
        x1, y1, x2, y2 = detection.xyxy[0]
        class_id = int(detection.cls[0])  # The class ID of the detected object

        if detection.id is None:
          object_id = len(previous_positions)
        else:
          object_id = int(detection.id) 

        if class_id in [1,2,3,4,6,7,8]: 
            velocity = get_object_velocity(object_id, x1, y1, x2, y2)
            vehicles.append({
                'bbox': (x1, y1, x2, y2),
                'velocity': velocity,
                'object_id': object_id
            })

    ans = "no"
    while ans!= "yes":
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                vehicle1 = vehicles[i]
                vehicle2 = vehicles[j]

                if check_collision(vehicle1['bbox'], vehicle2['bbox']):
                    if vehicle1['velocity'] < velocity_threshold and vehicle2['velocity'] < velocity_threshold:
                        print(f"Collision detected between vehicle {vehicle1['object_id']} and vehicle {vehicle2['object_id']}!")
                        log_collision(vehicle1['object_id'], vehicle2['object_id'], vehicle1['bbox'], vehicle2['bbox'])
                        send_email_notification(vehicle1['object_id'], vehicle2['object_id'], vehicle1['bbox'], vehicle2['bbox'])
                        ans = "yes"

    annotated_frame = results[0].plot()

    # Display the video with detected objects and bounding boxes
    cv2_imshow(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
