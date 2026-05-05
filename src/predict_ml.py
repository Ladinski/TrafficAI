# src/predict_ml.py

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


CSV_PATH = "data/traffic_log.csv"


def load_data():
    df = pd.read_csv(CSV_PATH, sep=";")
    df["start_time"] = pd.to_datetime(df["start_time"])
    return df


def aggregate_traffic(df):
    df["time_bin"] = df["start_time"].dt.floor("15min")

    traffic = df.groupby("time_bin").size().reset_index(name="vehicle_count")

    traffic["hour"] = traffic["time_bin"].dt.hour
    traffic["minute"] = traffic["time_bin"].dt.minute
    traffic["day_of_week"] = traffic["time_bin"].dt.dayofweek
    traffic["interval_15min"] = traffic["hour"] * 4 + traffic["minute"] // 15

    return traffic


def train_model(traffic):
    X = traffic[["hour", "minute", "day_of_week", "interval_15min"]]
    y = traffic["vehicle_count"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=10,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Model: Random Forest Regressor")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f} vehicles")
    print(f"R² Score: {r2_score(y_test, predictions):.2f}")

    return model


def predict_next_day(model, traffic):
    last_date = traffic["time_bin"].dt.date.max()
    next_day = pd.to_datetime(last_date) + pd.Timedelta(days=1)

    rows = []

    for interval in range(96):
        hour = interval // 4
        minute = (interval % 4) * 15

        timestamp = next_day + pd.Timedelta(hours=hour, minutes=minute)

        rows.append({
            "time": timestamp,
            "hour": hour,
            "minute": minute,
            "day_of_week": timestamp.dayofweek,
            "interval_15min": interval,
        })

    future = pd.DataFrame(rows)

    X_future = future[["hour", "minute", "day_of_week", "interval_15min"]]
    future["predicted_vehicle_count"] = model.predict(X_future)

    return future


def plot_prediction(future):
    plt.figure(figsize=(12, 6))
    plt.plot(future["time"], future["predicted_vehicle_count"])

    plt.title("Predicted Traffic for Next Day - Random Forest ML Model")
    plt.xlabel("Time")
    plt.ylabel("Predicted Number of Vehicles")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_prediction(future):
    future.to_csv("data/next_day_prediction_ml.csv", index=False)
    print("Saved prediction to data/next_day_prediction_ml.csv")


if __name__ == "__main__":
    df = load_data()
    traffic = aggregate_traffic(df)

    print(f"Loaded {len(df)} vehicle records")
    print(f"Created {len(traffic)} traffic intervals")

    model = train_model(traffic)

    future = predict_next_day(model, traffic)

    print(future[["time", "predicted_vehicle_count"]].head(10))

    save_prediction(future)
    plot_prediction(future)