# src/main.py

import cv2
from datetime import datetime

from src.detector import detect_cars
from src.tracker import Tracker
from src.utils import save_finished_tracks_to_csv


def main():
    cap = cv2.VideoCapture(1)  # 0 = default webcam, 1 = your phone

    if not cap.isOpened():
        print("Error: Could not open webcam. Is it connected?")
        return

    tracker = Tracker(
        max_distance=60.0,
        max_missed_frames=10,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        current_time = datetime.now()

        # 1) Detect cars in the frame
        detections = detect_cars(frame, conf_threshold=0.4)

        # 2) Update tracker with detections
        tracker.update(detections, current_time=current_time)

        # 3) Draw active tracks (cars) with IDs
        for track in tracker.active_tracks.values():
            x1, y1, x2, y2 = track.box
            car_id = track.id

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"Car {car_id}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # 4) Handle finished tracks
        finished = tracker.get_finished_tracks(clear=True)

        for car in finished:
            duration = car.duration_seconds
            print(
                f"[FINISHED] Car {car.id} | start: {car.start_time.time()} | "
                f"end: {car.last_seen_time.time()} | duration: {duration:.1f}s"
            )

        save_finished_tracks_to_csv(finished)

        # 5) Show the frame
        cv2.imshow("TrafficAI - Car Detection & Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
