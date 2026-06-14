# src/detector.py

from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# COCO vehicle classes
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def detect_cars(frame, conf_threshold: float = 0.4):
    """
    Detect vehicles in a frame.

    Returns:
    [
        {
            "box": (x1, y1, x2, y2),
            "conf": 0.87,
            "class_id": 2,
            "label": "car"
        }
    ]
    """

    results = model(frame, verbose=False)[0]

    detections = []

    if results.boxes is None:
        return detections

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    for box, cls_id, conf in zip(boxes, classes, confs):
        cls_id = int(cls_id)

        if conf < conf_threshold:
            continue

        if cls_id not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = box.astype(int)

        detections.append({
            "box": (x1, y1, x2, y2),
            "conf": float(conf),
            "class_id": cls_id,
            "label": VEHICLE_CLASSES[cls_id],
        })

    return detections