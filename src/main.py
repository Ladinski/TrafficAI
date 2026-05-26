# src/main.py

import cv2
import time
import numpy as np
from datetime import datetime

from src.detector import detect_cars
from src.tracker import Tracker
from src.utils import save_finished_tracks_to_csv


STREAM_URL = "https://s51.nysdot.skyvdn.com/rtplive/TA_035/chunklist_w728921246.m3u8"


ROI_POINTS = [
    (150, 110),  # top-left
    (190, 100),  # top-right
    (500, 200),  # bottom-right
    (370, 280),  # bottom-left
]


def open_stream():
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("Error: Could not open traffic stream.")
        return None

    print("✅ Connected to traffic stream.")
    return cap


def is_inside_roi(box):
    x1, y1, x2, y2 = box

    cx = float((x1 + x2) / 2)
    cy = float((y1 + y2) / 2)

    polygon = np.array(ROI_POINTS, dtype=np.int32)

    return cv2.pointPolygonTest(
        polygon,
        (cx, cy),
        False
    ) >= 0


def draw_roi(frame):
    polygon = np.array(ROI_POINTS, dtype=np.int32)

    cv2.polylines(
        frame,
        [polygon],
        isClosed=True,
        color=(0, 255, 255),
        thickness=2,
    )

    cv2.putText(
        frame,
        "Tracking Zone",
        ROI_POINTS[0],
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


def main():
    cap = open_stream()
    if cap is None:
        return

    tracker = Tracker(
        max_distance=60.0,
        max_missed_frames=10,
    )

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("⚠️ Stream read failed. Reconnecting...")
            cap.release()
            time.sleep(3)

            cap = open_stream()
            if cap is None:
                time.sleep(5)
                continue

            continue

        current_time = datetime.now()

        detections = detect_cars(frame, conf_threshold=0.4)
        detections = [d for d in detections if is_inside_roi(d["box"])]

        tracker.update(detections, current_time=current_time)

        draw_roi(frame)

        for track in tracker.active_tracks.values():
            x1, y1, x2, y2 = track.box

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Vehicle {track.id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        finished = tracker.get_finished_tracks(clear=True)

        for car in finished:
            print(
                f"[FINISHED] Car {car.id} | "
                f"start: {car.start_time.time()} | "
                f"end: {car.last_seen_time.time()} | "
                f"duration: {car.duration_seconds:.1f}s"
            )

        save_finished_tracks_to_csv(finished)

        cv2.imshow("TrafficAI - Live Traffic", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopping...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()