import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csv_auto(path):
    return pd.read_csv(path, sep=None, engine="python")


def load_actual_day(actual_csv, target_date):
    df = read_csv_auto(actual_csv)

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna(subset=["start_time"])

    day_start = pd.to_datetime(target_date).normalize()
    day_end = day_start + pd.Timedelta(days=1)

    df = df[(df["start_time"] >= day_start) & (df["start_time"] < day_end)]

    observed = (
        df.assign(time_bin=df["start_time"].dt.floor("15min"))
        .groupby("time_bin")
        .size()
        .reset_index(name="actual_vehicle_count")
    )

    full_index = pd.date_range(day_start, periods=96, freq="15min")

    actual = (
        observed.set_index("time_bin")
        .reindex(full_index)
        .rename_axis("time")
        .reset_index()
    )

    actual["is_observed"] = actual["actual_vehicle_count"].notna().astype(int)
    return actual


def load_prediction_day(prediction_csv, target_date):
    pred = read_csv_auto(prediction_csv)

    pred["time"] = pd.to_datetime(pred["time"], errors="coerce")
    pred = pred.dropna(subset=["time"])

    day_start = pd.to_datetime(target_date).normalize()
    day_end = day_start + pd.Timedelta(days=1)

    pred = pred[(pred["time"] >= day_start) & (pred["time"] < day_end)]

    full_index = pd.date_range(day_start, periods=96, freq="15min")

    pred = (
        pred.set_index("time")
        .reindex(full_index)
        .rename_axis("time")
        .reset_index()
    )

    return pred


def filter_time_range(data, start_time, end_time):
    start = pd.to_datetime(start_time).time()
    end = pd.to_datetime(end_time).time()

    return data[
        (data["time"].dt.time >= start)
        & (data["time"].dt.time <= end)
    ].copy()


def calculate_metrics(data):
    valid = (
        data["actual_vehicle_count"].notna()
        & data["predicted_vehicle_count"].notna()
    )

    if valid.sum() == 0:
        return np.nan, np.nan, 0

    error = (
        data.loc[valid, "predicted_vehicle_count"]
        - data.loc[valid, "actual_vehicle_count"]
    )

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    return mae, rmse, int(valid.sum())


def plot_comparison(data, target_date, start_time, end_time, output_path):
    mae, rmse, observed_points = calculate_metrics(data)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        data["time"],
        data["actual_vehicle_count"],
        marker="o",
        markersize=5,
        linewidth=2,
        label="Actual traffic",
    )

    ax.plot(
        data["time"],
        data["predicted_vehicle_count"],
        marker="o",
        markersize=5,
        linewidth=2,
        label="Predicted traffic",
    )

    date_label = pd.to_datetime(target_date).strftime("%d.%m.%Y")

    ax.set_title(
        f"Actual vs Predicted Traffic Volume "
        f"({date_label}, {start_time}–{end_time})"
    )
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Vehicle Count per 15-Minute Interval")

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax.set_xlim(data["time"].min(), data["time"].max())
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right")

    ax.text(
        0.01,
        -0.18,
        f"Observed intervals used: {observed_points} | "
        f"MAE: {mae:.2f} | RMSE: {rmse:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
    )

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare actual traffic logs with LSTM predictions."
    )

    parser.add_argument(
        "--actual-csv",
        default="data/traffic_log3.csv",
        help="Path to actual traffic log CSV.",
    )

    parser.add_argument(
        "--prediction-csv",
        default="data/next_day_prediction_lstm.csv",
        help="Path to prediction CSV.",
    )

    parser.add_argument(
        "--date",
        default="2026-05-18",
        help="Target date.",
    )

    parser.add_argument(
        "--start-time",
        default="14:00",
        help="Comparison start time.",
    )

    parser.add_argument(
        "--end-time",
        default="23:45",
        help="Comparison end time.",
    )

    parser.add_argument(
        "--output-dir",
        default="comparison_output",
        help="Output folder.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual = load_actual_day(args.actual_csv, args.date)
    predicted = load_prediction_day(args.prediction_csv, args.date)

    comparison = actual.merge(predicted, on="time", how="left")

    comparison = filter_time_range(
        comparison,
        args.start_time,
        args.end_time,
    )

    comparison["error"] = (
        comparison["predicted_vehicle_count"]
        - comparison["actual_vehicle_count"]
    )
    comparison["absolute_error"] = comparison["error"].abs()

    comparison_csv = output_dir / "actual_vs_predicted_observed_period.csv"
    graph_path = output_dir / "actual_vs_predicted_observed_period.png"

    comparison.to_csv(comparison_csv, index=False)

    plot_comparison(
        comparison,
        args.date,
        args.start_time,
        args.end_time,
        graph_path,
    )

    mae, rmse, observed_points = calculate_metrics(comparison)

    print(f"Saved comparison CSV to: {comparison_csv}")
    print(f"Saved graph to: {graph_path}")
    print(f"Observed intervals used: {observed_points}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")


if __name__ == "__main__":
    main()