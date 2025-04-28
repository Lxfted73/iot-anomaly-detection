# IoT Anomaly Detection Project

This project implements a real-time anomaly detection system for IoT sensor data using the UCI Air Quality dataset. The pipeline leverages an autoencoder-based approach, enhanced with Isolation Forest and One-Class SVM for robust anomaly detection. The system processes sensor data, handles missing values, engineers temporal features, and evaluates performance on simulated anomalies.

## Dataset
- **Source**: [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
- **Description**: Contains hourly air quality measurements (e.g., CO, NO2, NOx, C6H6) from a multi-sensor device in an Italian city, with 9357 samples and simulated anomalies (2.99%).
- **File**: `data/AirQualityUCI.csv`

## Features
- **Preprocessing**: Handles missing values using KNN imputation and normalizes data with RobustScaler.
- **Feature Engineering**: Includes derived features like rolling means, variances, differences, CO/NO2 ratio, hourly deviations, and a night score for temporal patterns.
- **Model**: Autoencoder with increased capacity (128-64-32-16 layers), combined with an ensemble of Isolation Forest and One-Class SVM.
- **Thresholding**: Dynamic thresholding with feature-weighted MSE, tuned on a validation set.
- **Evaluation**: Cross-validation and final metrics (precision, recall, F1-score) on simulated anomalies.

## Results
- **Final Metrics** (Simulated Data):
  - Precision: 0.68,
  - Recall: 0.50, 
  - F1-Score: 0.58
- **Key Features**: `NO2_rolling_var`, `CO_diff`, `CO_rolling_var` are the top contributors to anomaly detection.

## Directory Structure
