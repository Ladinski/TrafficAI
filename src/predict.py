import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "data/traffic_log.csv"


def load_data():
    df = pd.read_csv(CSV_PATH, sep=";")

    
    df["start_time"] = pd.to_datetime(df["start_time"])

    return df


def aggregate_traffic(df):
    
    df["interval"] = df["start_time"].dt.floor("15min")

    traffic_counts = df.groupby("interval").size().reset_index(name="car_count")

    return traffic_counts


def plot_traffic(traffic_counts):
    plt.figure(figsize=(12, 6))
    plt.plot(traffic_counts["interval"], traffic_counts["car_count"])
    plt.title("Traffic Volume Over Time")
    plt.xlabel("Time")
    plt.ylabel("Number of Vehicles")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def predict_next_day(traffic_counts):
    traffic_counts["interval_15min"] = (
        traffic_counts["interval"].dt.hour * 4 +
        traffic_counts["interval"].dt.minute // 15
    )

    
    full_range = pd.DataFrame({"interval_15min": range(96)})

    avg_pattern = (
        traffic_counts.groupby("interval_15min")["car_count"]
        .mean()
        .reset_index()
    )

    avg_pattern = full_range.merge(avg_pattern, on="interval_15min", how="left")

    avg_pattern["car_count"] = avg_pattern["car_count"].ffill()

    return avg_pattern


def plot_prediction(avg_pattern):
    plt.figure(figsize=(12, 6))
    plt.plot(avg_pattern["interval_15min"], avg_pattern["car_count"])
    plt.title("Predicted Traffic Pattern (Next Day)")
    plt.xlabel("15-min Interval (0–95)")
    plt.ylabel("Expected Vehicles")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    traffic = aggregate_traffic(df)

    print("Plotting historical traffic...")
    plot_traffic(traffic)

    prediction = predict_next_day(traffic)

    print("Plotting prediction...")
    plot_prediction(prediction)