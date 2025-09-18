import os
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
import time
from plyer import notification
import winsound


os.makedirs("collision_frames", exist_ok=True)
LOG_FILE = "collision_log.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["time", "vehicle1", "vehicle2", "bbox1", "bbox2", "frame_path"]).to_csv(LOG_FILE, index=False)

model = YOLO('yolov8m.pt')

video_path = './video3.mp4'
cap = cv2.VideoCapture(video_path)

velocity_threshold = 1.5 
previous_positions = {}
notified_collisions = set()
notified_collisions = {}
cooldown_time = 600
frame_count = 0

def get_object_velocity(object_id, x1, y1, x2, y2):
    current_center = np.array([
        (x1.item() + x2.item()) / 2,  
        (y1.item() + y2.item()) / 2   
    ])

    if object_id in previous_positions:
        previous_center = previous_positions[object_id]
        distance = np.linalg.norm(current_center - previous_center)
        velocity = distance
    else:
        velocity = 0
    previous_positions[object_id] = current_center
    return velocity


def log_collision(vehicle1_id, vehicle2_id, bbox1, bbox2, frame):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    frame_filename = f"collision_frames/{current_time.replace(':','-').replace(' ','_')}.jpg"
    cv2.imwrite(frame_filename, frame)
    new_log = pd.DataFrame([{
        "time": current_time,
        "vehicle1": int(vehicle1_id),
        "vehicle2": int(vehicle2_id),
        "bbox1": str(bbox1),
        "bbox2": str(bbox2),
        "frame_path": frame_filename
    }], columns=["time", "vehicle1", "vehicle2", "bbox1", "bbox2", "frame_path"])
    new_log.to_csv(LOG_FILE, mode="a", header=False, index=False)

    log_message = (f"Collision detected between Vehicle {vehicle1_id} and Vehicle {vehicle2_id} at {current_time}.\n"
                   f"Vehicle 1 bounding box: {bbox1}\n"
                   f"Vehicle 2 bounding box: {bbox2}\n\n")
    with open('collision_log.txt', 'a') as log_file:
        log_file.write(log_message)

    print(f"[{current_time}] Collision logged between {vehicle1_id} and {vehicle2_id}")


def notify_user(vehicle1_id, vehicle2_id):
    notification.notify(
        title="COLLISION ALERT!",
        message=f"Collision detected between Vehicle {vehicle1_id} and Vehicle {vehicle2_id}",
        timeout=5
    )
    winsound.Beep(1000, 700)
    

def check_collision(bbox1, bbox2, iou_threshold=0.3):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return False  
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area
    return iou >= iou_threshold


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    frame_count += 1
    results = model.track(frame, persist=True)
    vehicles = []

    for detection in results[0].boxes:
        x1, y1, x2, y2 = detection.xyxy[0]
        class_id = int(detection.cls[0])  
        object_id = int(detection.id.item()) if detection.id is not None else hash((frame_count, x1.item(), y1.item(), x2.item(), y2.item()))

        if class_id in [1,2,3,4,6,7,8]: 
            velocity = get_object_velocity(object_id, x1, y1, x2, y2)
            vehicles.append({
                'bbox': (x1.item(), y1.item(), x2.item(), y2.item()),
                'velocity': velocity,
                'object_id': object_id
            })

    min_movement = 0.5
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            v1, v2 = vehicles[i], vehicles[j]
            if check_collision(v1['bbox'], v2['bbox']):
                if (v1['velocity'] > min_movement or v2['velocity'] > min_movement):
                    pair = tuple(sorted([v1['object_id'], v2['object_id']]))
                    current_time = time.time()

                    if pair not in notified_collisions or (current_time - notified_collisions[pair]) > cooldown_time:
                        print(f"Collision detected between vehicle {v1['object_id']} and {v2['object_id']}!")
                        log_collision(v1['object_id'], v2['object_id'], v1['bbox'], v2['bbox'], frame)
                        notify_user(v1['object_id'], v2['object_id'])
                        notified_collisions[pair] = current_time

    annotated_frame = results[0].plot()
    cv2.imshow("Collision Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
