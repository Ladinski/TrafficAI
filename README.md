# Traffic AI 🚦

**Intelligent Traffic Monitoring and Prediction System Using Computer Vision and Machine Learning**

Traffic AI is a capstone project that combines Computer Vision, Deep Learning, and Time-Series Forecasting to analyze live traffic camera feeds, monitor vehicle activity in real time, and predict future traffic conditions.

The system uses YOLOv8 for vehicle detection, a custom centroid-based tracking algorithm for vehicle counting, and an LSTM neural network for next-day traffic forecasting.

---

## Overview

Traffic congestion remains a major challenge for modern cities, leading to delays, increased fuel consumption, and environmental pollution. Traditional traffic monitoring solutions often require expensive infrastructure such as embedded sensors, radar systems, and specialized hardware.

Traffic AI demonstrates how publicly available traffic camera feeds can be transformed into intelligent traffic analytics using Artificial Intelligence and Machine Learning without requiring additional physical infrastructure.

---

## Features

### Real-Time Vehicle Detection

* YOLOv8-based vehicle detection
* Supports:

  * Cars
  * Trucks
  * Vans
  * Buses
* Real-time video processing using OpenCV

### Vehicle Tracking

* Custom centroid-based multi-object tracking
* Persistent vehicle IDs
* Duplicate count prevention
* Vehicle counting across video frames

### Traffic Analytics

* Traffic density estimation
* Vehicle count aggregation
* Historical traffic pattern generation
* CSV-based logging system

### Traffic Forecasting

* LSTM neural network
* 24-hour traffic prediction
* Direct multi-step forecasting
* Time-series preprocessing and feature engineering

---

## System Architecture

```text
Traffic Camera Feed
        ↓
OpenCV Video Processing
        ↓
YOLOv8 Vehicle Detection
        ↓
Centroid-Based Tracking
        ↓
Traffic Analytics Generation
        ↓
CSV Data Storage
        ↓
LSTM Forecasting Model
        ↓
Traffic Predictions
```

---

## Technology Stack

| Technology | Purpose              |
| ---------- | -------------------- |
| Python     | Core development     |
| OpenCV     | Video processing     |
| YOLOv8     | Vehicle detection    |
| PyTorch    | Deep learning        |
| NumPy      | Numerical computing  |
| Pandas     | Data processing      |
| Roboflow   | Dataset management   |
| CSV        | Traffic data storage |

---

## Dataset

The detection model was trained on approximately **9,800 annotated traffic images** collected from publicly available Roboflow traffic datasets.

Dataset characteristics:

* Highway traffic
* Urban traffic scenes
* Various weather conditions
* Multiple lighting environments
* Different traffic densities

---

## Vehicle Detection Pipeline

1. Capture video frame
2. Resize frame
3. Run YOLOv8 inference
4. Extract detections
5. Filter vehicle classes
6. Draw bounding boxes
7. Send detections to tracking system

---

## Traffic Forecasting

Traffic data is aggregated into 15-minute intervals and used to train a Long Short-Term Memory (LSTM) neural network.

### Forecasting Features

* Vehicle count
* Observation flag
* Time-of-day encoding
* Day-of-week encoding

### Model Configuration

* 2 stacked LSTM layers
* Hidden size: 64 neurons
* Dropout regularization
* 24 historical intervals as input
* 96 future intervals as output

### Evaluation Metrics

* Mean Absolute Error (MAE): 307.73 vehicles
* Root Mean Squared Error (RMSE): 381.03 vehicles

---

## Results

The system successfully demonstrated:

* Real-time vehicle detection
* Stable vehicle tracking
* Traffic density estimation
* Historical traffic analytics
* Next-day traffic forecasting

The project shows that intelligent traffic monitoring can be achieved using existing camera infrastructure and consumer-grade hardware.

---

## Future Improvements

* DeepSORT integration
* ByteTrack integration
* Multi-camera support
* Cloud deployment
* Real-time web dashboard
* Weather-aware forecasting
* Macedonia traffic camera integration
* Smart city deployment

---

## Hardware Used

* AMD Ryzen 5 5500H
* NVIDIA RTX 2050
* 16 GB RAM

---

## Author

**Borjan Ladinski**

Capstone Project
Faculty of Computer Science and Information Technology

---

## License

This project is intended for educational and research purposes.
