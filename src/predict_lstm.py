import argparse
import os
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CSV_PATH = "data/traffic_log2.csv"
PREDICTION_PATH = "data/next_day_prediction_lstm.csv"
MODEL_PATH = "models/traffic_lstm.pt"


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values):
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std == 0] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values):
        return (values - self.mean) / self.std

    def inverse_target(self, values):
        return values * self.std[0] + self.mean[0]


class TrafficLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        output_size=96,
        hidden_size=64,
        num_layers=2,
        dropout=0.15,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


def load_data(csv_path):
    df = pd.read_csv(csv_path, sep=";")
    df["start_time"] = pd.to_datetime(df["start_time"])
    return df


def aggregate_traffic(df):
    observed_traffic = (
        df.assign(time_bin=df["start_time"].dt.floor("15min"))
        .groupby("time_bin")
        .size()
        .reset_index(name="vehicle_count")
        .sort_values("time_bin")
    )

    full_index = pd.date_range(
        start=observed_traffic["time_bin"].min(),
        end=observed_traffic["time_bin"].max(),
        freq="15min",
    )

    traffic = (
        observed_traffic.set_index("time_bin")
        .reindex(full_index)
        .rename_axis("time_bin")
        .reset_index()
    )

    traffic["is_observed"] = traffic["vehicle_count"].notna().astype(float)

    traffic["hour"] = traffic["time_bin"].dt.hour
    traffic["minute"] = traffic["time_bin"].dt.minute
    traffic["day_of_week"] = traffic["time_bin"].dt.dayofweek
    traffic["interval_15min"] = traffic["hour"] * 4 + traffic["minute"] // 15

    traffic["interval_sin"] = np.sin(2 * np.pi * traffic["interval_15min"] / 96)
    traffic["interval_cos"] = np.cos(2 * np.pi * traffic["interval_15min"] / 96)
    traffic["day_sin"] = np.sin(2 * np.pi * traffic["day_of_week"] / 7)
    traffic["day_cos"] = np.cos(2 * np.pi * traffic["day_of_week"] / 7)

    interval_pattern = (
        observed_traffic.assign(
            interval_15min=(
                observed_traffic["time_bin"].dt.hour * 4
                + observed_traffic["time_bin"].dt.minute // 15
            )
        )
        .groupby("interval_15min")["vehicle_count"]
        .median()
    )
    fallback_count = observed_traffic["vehicle_count"].median()
    traffic["filled_vehicle_count"] = traffic["vehicle_count"].fillna(
        traffic["interval_15min"].map(interval_pattern)
    )
    traffic["filled_vehicle_count"] = traffic["filled_vehicle_count"].fillna(
        fallback_count
    )

    return traffic


def build_sequences(traffic, sequence_length, prediction_horizon, min_observed_ratio):
    feature_columns = [
        "filled_vehicle_count",
        "is_observed",
        "interval_sin",
        "interval_cos",
        "day_sin",
        "day_cos",
    ]
    values = traffic[feature_columns].to_numpy(dtype=np.float32)
    scaler = StandardScaler.fit(values)
    scaled = scaler.transform(values).astype(np.float32)
    observed = traffic["is_observed"].to_numpy(dtype=np.float32)

    x_rows = []
    y_rows = []
    mask_rows = []
    target_start_times = []

    max_start = len(scaled) - prediction_horizon + 1
    for index in range(sequence_length, max_start):
        past_observed_ratio = observed[index - sequence_length:index].mean()
        future_observed_ratio = observed[index:index + prediction_horizon].mean()
        if (
            past_observed_ratio < min_observed_ratio
            or future_observed_ratio < min_observed_ratio
        ):
            continue

        x_rows.append(scaled[index - sequence_length:index])
        y_rows.append(scaled[index:index + prediction_horizon, 0])
        mask_rows.append(observed[index:index + prediction_horizon])
        target_start_times.append(traffic["time_bin"].iloc[index])

    if not x_rows:
        raise ValueError(
            "Not enough observed traffic windows. Lower --min-observed-ratio, "
            "reduce --sequence-length/--prediction-horizon, or collect more continuous data."
        )

    x = torch.tensor(np.array(x_rows), dtype=torch.float32)
    y = torch.tensor(np.array(y_rows), dtype=torch.float32)
    masks = torch.tensor(np.array(mask_rows), dtype=torch.float32)
    return x, y, masks, scaler, feature_columns, target_start_times


def split_train_test(x, y, masks, test_size):
    split_index = max(1, int(len(x) * (1 - test_size)))
    if split_index >= len(x):
        split_index = len(x) - 1
    return (
        x[:split_index],
        y[:split_index],
        masks[:split_index],
        x[split_index:],
        y[split_index:],
        masks[split_index:],
    )


def train_model(
    x,
    y,
    epochs,
    batch_size,
    learning_rate,
    hidden_size,
    num_layers,
    prediction_horizon,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TrafficLSTM(
        input_size=x.shape[-1],
        output_size=prediction_horizon,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs} - loss: {np.mean(losses):.4f}")

    return model, device


def evaluate_model(model, device, x_test, y_test, mask_test, scaler):
    if len(x_test) == 0:
        print("Test set is empty; skipping evaluation.")
        return

    model.eval()
    with torch.no_grad():
        predictions = model(x_test.to(device)).cpu().numpy()

    actual = scaler.inverse_target(y_test.numpy())
    predicted = scaler.inverse_target(predictions)
    observed_mask = mask_test.numpy().astype(bool)

    observed_actual = actual[observed_mask]
    observed_predicted = predicted[observed_mask]
    if len(observed_actual) == 0:
        print("No observed target points in test set; skipping metrics.")
        return

    errors = observed_predicted - observed_actual
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    print("Test evaluation on hidden observed intervals")
    print(f"MAE: {mae:.2f} vehicles")
    print(f"RMSE: {rmse:.2f} vehicles")


def predict_next_day(model, device, traffic, scaler, sequence_length, prediction_horizon):
    model.eval()
    last_time = traffic["time_bin"].max()
    next_day = (last_time.normalize() + pd.Timedelta(days=1)).to_pydatetime()
    future_times = pd.date_range(start=next_day, periods=prediction_horizon, freq="15min")

    history = traffic[
        [
            "filled_vehicle_count",
            "is_observed",
            "interval_sin",
            "interval_cos",
            "day_sin",
            "day_cos",
        ]
    ].to_numpy(dtype=np.float32)
    scaled_history = scaler.transform(history).astype(np.float32)

    with torch.no_grad():
        input_window = torch.tensor(
            np.array([scaled_history[-sequence_length:]], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        scaled_predictions = model(input_window).cpu().numpy()[0]

    vehicle_counts = np.maximum(0.0, scaler.inverse_target(scaled_predictions))
    predictions = [
        {
            "time": timestamp,
            "predicted_vehicle_count": round(float(vehicle_count), 2),
        }
        for timestamp, vehicle_count in zip(future_times, vehicle_counts)
    ]

    return pd.DataFrame(predictions)


def save_model(model, scaler, sequence_length, prediction_horizon, feature_columns, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
            "sequence_length": sequence_length,
            "prediction_horizon": prediction_horizon,
            "feature_columns": feature_columns,
        },
        path,
    )
    print(f"Saved LSTM model to {path}")


def save_prediction(future, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    future.to_csv(path, index=False)
    print(f"Saved LSTM prediction to {path}")


def plot_prediction(future, plot_type):
    fig, ax = plt.subplots(figsize=(12, 6))

    if plot_type == "bar":
        counts = future["predicted_vehicle_count"]
        low_limit = counts.quantile(0.33)
        high_limit = counts.quantile(0.66)
        colors = np.select(
            [counts <= low_limit, counts >= high_limit],
            ["#86c5da", "#1f5a99"],
            default="#3f8fc5",
        )
        ax.bar(
            future["time"],
            counts,
            width=0.008,
            color=colors,
            edgecolor="#0f3f6f",
            linewidth=0.4,
        )
        legend_handles = [
            mpatches.Patch(color="#86c5da", label="Low"),
            mpatches.Patch(color="#3f8fc5", label="Medium"),
            mpatches.Patch(color="#1f5a99", label="High"),
        ]
        ax.legend(handles=legend_handles, title="Traffic", loc="upper right")
    else:
        ax.plot(future["time"], future["predicted_vehicle_count"], color="#1f77b4")

    ax.set_xlim(future["time"].min(), future["time"].max())
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_fontsize(8)
        label.set_ha("center")

    prediction_date = future["time"].min().strftime("%d.%m.%Y")
    plt.title(
        f"Predicted Traffic for Next Day ({prediction_date}) - LSTM Deep Learning Model"
    )
    plt.xlabel("Time")
    plt.ylabel("Predicted Number of Vehicles")
    plt.grid(True)
    fig.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an LSTM model and predict next-day traffic volume."
    )
    parser.add_argument("--csv", default=CSV_PATH, help="Input traffic CSV path.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=24,
        help="Number of past 15-minute bins used by the LSTM.",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=96,
        help="Number of future 15-minute bins to predict.",
    )
    parser.add_argument(
        "--min-observed-ratio",
        type=float,
        default=0.75,
        help="Minimum real-data ratio required in each training window.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Latest fraction of training windows hidden for evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--output", default=PREDICTION_PATH)
    parser.add_argument("--model-output", default=MODEL_PATH)
    parser.add_argument(
        "--plot-type",
        choices=["bar", "line"],
        default="bar",
        help="Use 96 interval bars or a continuous line plot.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot.")
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_data(args.csv)
    traffic = aggregate_traffic(df)
    x, y, masks, scaler, feature_columns, target_start_times = build_sequences(
        traffic=traffic,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        min_observed_ratio=args.min_observed_ratio,
    )
    x_train, y_train, mask_train, x_test, y_test, mask_test = split_train_test(
        x, y, masks, args.test_size
    )

    print(f"Loaded {len(df)} vehicle records from {args.csv}")
    print(f"Created {len(traffic)} regular 15-minute traffic intervals")
    print(f"Observed intervals: {int(traffic['is_observed'].sum())}/{len(traffic)}")
    print(
        "Training direct LSTM: "
        f"{args.sequence_length} past intervals -> {args.prediction_horizon} future intervals"
    )
    print(f"Usable windows after gap filtering: {len(x)}")
    print(f"Train windows: {len(x_train)} | Test windows: {len(x_test)}")
    if target_start_times:
        print(
            "Test windows are the latest hidden part of your existing data, "
            "not tomorrow's unknown traffic."
        )

    model, device = train_model(
        x=x_train,
        y=y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        prediction_horizon=args.prediction_horizon,
    )
    evaluate_model(model, device, x_test, y_test, mask_test, scaler)

    future = predict_next_day(
        model=model,
        device=device,
        traffic=traffic,
        scaler=scaler,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
    )

    print(future.head(10))
    save_prediction(future, args.output)
    save_model(
        model,
        scaler,
        args.sequence_length,
        args.prediction_horizon,
        feature_columns,
        args.model_output,
    )

    if not args.no_plot:
        plot_prediction(future, args.plot_type)


if __name__ == "__main__":
    main()
