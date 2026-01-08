# src/utils.py

import os
import pandas as pd

# Base directory = project root (one level above src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "traffic_log.csv")

COLUMNS = [
    "car_id",
    "start_time",
    "end_time",
    "duration_seconds",
]

# Minimum duration (in seconds) for a car to be considered valid
MIN_DURATION_SECONDS = 0.5


def save_finished_tracks_to_csv(tracks):
    """
    Append finished car tracks to the CSV log file.
    Creates the file (and folder) if it doesn't exist yet.
    Filters out cars with too-short durations.
    """
    # Filter out invalid / too short tracks
    valid_tracks = [
        car for car in tracks
        if car.duration_seconds >= MIN_DURATION_SECONDS
    ]

    if not valid_tracks:
        return

    # Make sure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    rows = []
    for car in valid_tracks:
        rows.append({
            "car_id": car.id,
            "start_time": car.start_time,
            "end_time": car.last_seen_time,
            "duration_seconds": car.duration_seconds,
        })

    df_new = pd.DataFrame(rows, columns=COLUMNS)

    if os.path.exists(CSV_PATH):
        df_existing = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(CSV_PATH, index=False, sep=";", encoding="utf-8")
    else:
        df_new.to_csv(CSV_PATH, index=False, sep=";", encoding="utf-8")

    print(f"💾 Saved {len(valid_tracks)} car(s) to {CSV_PATH}")
