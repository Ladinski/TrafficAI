# src/detector.py

from ultralytics import YOLO
import numpy as np

# Load YOLO model once when this module is imported.
# "yolov8n.pt" = nano model (small & fast). Good for real-time.
model = YOLO("yolov8n.pt")

# COCO class index for 'car' is 2.
CAR_CLASS_ID = 2

def detect_cars(frame, conf_threshold: float = 0.4):
    """
    Run 0YOLO on the given frame and return a list of bounding boxes for cars.

    :param frame: BGR image from OpenCV (numpy array)
    :param conf_threshold: minimum confidence to accept a detection
    :return: list of dicts like:
             [{"box": (x1, y1, x2, y2), "conf": 0.87}, ...]
    """
    # Run inference. YOLO accepts OpenCV's BGR numpy arrays directly.
    results = model(frame, verbose=False)[0]  # first (and only) image in batch

    detections = []

    if results.boxes is None:
        return detections

    boxes = results.boxes.xyxy.cpu().numpy()   # shape: [N, 4]
    classes = results.boxes.cls.cpu().numpy()  # shape: [N]
    confs = results.boxes.conf.cpu().numpy()   # shape: [N]

    for box, cls_id, conf in zip(boxes, classes, confs):
        if conf < conf_threshold:
            continue

        # Only keep cars for now (class id 2).
        # Later you can add trucks/buses if you want.
        if int(cls_id) != CAR_CLASS_ID:
            continue

        x1, y1, x2, y2 = box.astype(int)
        detections.append({
            "box": (x1, y1, x2, y2),
            "conf": float(conf)
        })

    return detections
