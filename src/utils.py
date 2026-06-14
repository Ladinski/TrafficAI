

import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "traffic_log.csv")

COLUMNS = [
    "car_id",
    "start_time",
    "end_time",
    "duration_seconds",
    "date",
    "day_of_week_name",
    "day_of_week_num",
    "hour",
    "interval_15min",
]


MIN_DURATION_SECONDS = 0.5


def save_finished_tracks_to_csv(tracks):
    """
    Append finished car tracks to the CSV log file (semicolon-separated).
    Creates the file (and folder) if it doesn't exist yet.
    Filters out cars with too-short durations.
    """
    
    valid_tracks = [car for car in tracks if car.duration_seconds >= MIN_DURATION_SECONDS]
    if not valid_tracks:
        return

  
    os.makedirs(DATA_DIR, exist_ok=True)

    rows = []
    for car in valid_tracks:
        start_dt = car.start_time
        end_dt = car.last_seen_time

        rows.append({
            "car_id": car.id,

      
            "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(car.duration_seconds, 2),

            
            "date": start_dt.strftime("%Y-%m-%d"),
            "day_of_week_name": start_dt.strftime("%A"),  
            "day_of_week_num": start_dt.weekday(),         
            "hour": start_dt.hour,                         
            "interval_15min": (start_dt.hour * 60 + start_dt.minute) // 15,  
        })

    df_new = pd.DataFrame(rows, columns=COLUMNS)

    if os.path.exists(CSV_PATH):
        df_existing = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(CSV_PATH, index=False, sep=";", encoding="utf-8")
    else:
        df_new.to_csv(CSV_PATH, index=False, sep=";", encoding="utf-8")

    print(f"Saved {len(df_new)} car(s) to {CSV_PATH}")
