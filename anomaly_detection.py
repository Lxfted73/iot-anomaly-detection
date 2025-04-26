#!/usr/bin/env python
# coding: utf-8

# # IoT Anomaly Detection Project
#
# This notebook implements real-time anomaly detection for IoT sensor data using the UCI Air Quality dataset.
# The pipeline includes data analysis, preprocessing, Autoencoder (AE) training, evaluation with simulated ground truth, and visualization.
#
# **Dataset**: [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
# **Tools**: Pandas, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, NumPy, Pickle, Imblearn
# **Outputs**:
# - `preprocessed_data.csv`, `simulated_data.csv`, `anomaly_predictions_simulated.csv`
# - `correlation_matrix.png`, `feature_distributions.png`, `diurnal_patterns.png`, `confusion_matrix_simulated.png`, `mse_distribution.png`
# - `ae.keras`, `scaler.pkl`

# ## Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ## Data Analysis
# Load data
df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')
df['Date_Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')

# Select sensor columns
sensor_cols = ['CO(GT)', 'NO2(GT)', 'NOx(GT)', 'C6H6(GT)', 'NMHC(GT)']

# Summary statistics
print("Summary Statistics:")
print(df[sensor_cols].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]))

# Missing values (-200)
missing_counts = (df[sensor_cols] == -200).sum()
missing_percent = (missing_counts / len(df)) * 100
print("\nMissing Values (-200):")
print(pd.DataFrame({'Count': missing_counts, 'Percentage': missing_percent}))

# Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df[sensor_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Sensor Features')
plt.savefig('correlation_matrix.png')
plt.close()

# Feature distributions
plt.figure(figsize=(15, 10))
for i, col in enumerate(sensor_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

# Diurnal patterns
df['Hour'] = df['Date_Time'].dt.hour
hourly_means = df.groupby('Hour')[sensor_cols].mean()
plt.figure(figsize=(12, 8))
for col in sensor_cols:
    plt.plot(hourly_means.index, hourly_means[col], label=col)
plt.title('Diurnal Patterns (Hourly Means)')
plt.xlabel('Hour of Day')
plt.ylabel('Mean Value')
plt.legend()
plt.savefig('diurnal_patterns.png')
plt.close()

# ## Data Preprocessing
# Drop rows with NaN Date_Time
df = df.dropna(subset=['Date_Time']).reset_index(drop=True)
logger.info(f"Dropped {9357 - len(df)} rows with NaN Date_Time. New dataset size: {len(df)}")

# Replace -200 with NaN and impute
df[sensor_cols] = df[sensor_cols].replace(-200, np.nan)
df[sensor_cols] = df[sensor_cols].interpolate(method='linear', limit_direction='both')

# Compute derived features
df['CO_rolling_mean'] = df['CO(GT)'].rolling(window=3, min_periods=1).mean()
df['NO2_rolling_mean'] = df['NO2(GT)'].rolling(window=3, min_periods=1).mean()
df['C6H6_rolling_mean'] = df['C6H6(GT)'].rolling(window=3, min_periods=1).mean()
df['NOx_lag1'] = df['NOx(GT)'].shift(1).bfill()

# Add change-rate features
df['CO_diff'] = df['CO(GT)'].diff().fillna(0)
df['NO2_diff'] = df['NO2(GT)'].diff().fillna(0)

# Define features for modeling
feature_cols = ['CO(GT)', 'NO2(GT)', 'NOx(GT)', 'C6H6(GT)', 'CO_rolling_mean',
                'NO2_rolling_mean', 'C6H6_rolling_mean', 'NOx_lag1', 'CO_diff', 'NO2_diff']
if not all(col in df.columns for col in feature_cols):
    raise ValueError("Missing feature columns in DataFrame")

# Impute any remaining NaN in feature_cols
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

# Normalize features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[feature_cols])

# Check for NaN in scaled data
if np.any(np.isnan(data_scaled)):
    raise ValueError("NaN values found in scaled data.")

# Save preprocessed data
df.to_csv('preprocessed_data.csv', index=False)

# ## Autoencoder Training and Anomaly Detection
# Create simulated data with diverse anomalies (3%)
df_simulated = df.copy()
anomaly_indices = np.random.choice(len(df), size=int(0.03 * len(df)), replace=False)
for idx in anomaly_indices:
    r = np.random.random()
    if r < 0.2:  # Combined CO/NO2/NOx Spike
        df_simulated.loc[idx, 'CO(GT)'] = np.random.uniform(20, 25)
        df_simulated.loc[idx, 'NO2(GT)'] = np.random.uniform(450, 500)
        df_simulated.loc[idx, 'NOx(GT)'] = np.random.uniform(1200, 1400)
    elif r < 0.4:  # Rapid CO/NO2 Change
        df_simulated.loc[idx, 'NO2(GT)'] = np.random.uniform(350, 400)
        df_simulated.loc[idx, 'CO(GT)'] = np.random.uniform(15, 18)
    elif r < 0.6:  # Sustained High CO/NO2
        df_simulated.loc[idx:idx+2, 'CO(GT)'] = np.random.uniform(16, 20)
        df_simulated.loc[idx:idx+2, 'NO2(GT)'] = np.random.uniform(400, 450)
    elif r < 0.8:  # High Benzene + NOx
        df_simulated.loc[idx, 'C6H6(GT)'] = np.random.uniform(80, 100)
        df_simulated.loc[idx, 'NOx(GT)'] = np.random.uniform(800, 1000)
    else:  # Nighttime CO/NO2 Spike
        if df_simulated.loc[idx, 'Date_Time'].hour in [0, 1, 2, 3, 22, 23]:
            df_simulated.loc[idx, 'CO(GT)'] = np.random.uniform(12, 15)
            df_simulated.loc[idx, 'NO2(GT)'] = np.random.uniform(300, 350)

# Compute derived features for simulated data
df_simulated['CO_rolling_mean'] = df_simulated['CO(GT)'].rolling(window=3, min_periods=1).mean()
df_simulated['NO2_rolling_mean'] = df_simulated['NO2(GT)'].rolling(window=3, min_periods=1).mean()
df_simulated['C6H6_rolling_mean'] = df_simulated['C6H6(GT)'].rolling(window=3, min_periods=1).mean()
df_simulated['NOx_lag1'] = df_simulated['NOx(GT)'].shift(1).bfill()
df_simulated['CO_diff'] = df_simulated['CO(GT)'].diff().fillna(0)
df_simulated['NO2_diff'] = df_simulated['NO2(GT)'].diff().fillna(0)

# Impute any remaining NaN in feature_cols for simulated data
df_simulated[feature_cols] = df_simulated[feature_cols].fillna(df_simulated[feature_cols].mean())

# Normalize simulated data
data_scaled_simulated = scaler.transform(df_simulated[feature_cols])

# Check for NaN in scaled simulated data
if np.any(np.isnan(data_scaled_simulated)):
    raise ValueError("NaN values found in scaled simulated data.")

# Save simulated data
df_simulated.to_csv('simulated_data.csv', index=False)

# Split simulated data into training and validation sets
X_train_sim, X_val_sim, sim_train_indices, sim_val_indices = train_test_split(
    data_scaled_simulated, np.arange(len(data_scaled_simulated)), test_size=0.2, random_state=42
)
logger.info("Simulated data split into training and validation sets.")

# Define AE for simulated data
input_dim = len(feature_cols)
latent_dim = 32
l2_reg = 0.005
inputs = Input(shape=(input_dim,))
h = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(inputs)
h = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(h)
h = Dense(16, activation='relu', kernel_regularizer=l2(l2_reg))(h)
h = Dropout(0.2)(h)
latent = Dense(latent_dim, activation='relu')(h)
decoder_h = Dense(16, activation='relu', kernel_regularizer=l2(l2_reg))(latent)
decoder_h = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(decoder_h)
decoder_h = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(decoder_h)
decoder_h = Dropout(0.2)(decoder_h)
outputs = Dense(input_dim, activation='linear')(decoder_h)
ae_simulated = Model(inputs, outputs)
ae_simulated.compile(optimizer='adam', loss='mse')
logger.info("AE model for simulated data created.")

# Identify normal and anomalous data for training
key_features = ['NO2(GT)', 'C6H6(GT)']
feature_indices = [feature_cols.index(feat) for feat in key_features]
percentile_threshold = 97
anomaly_mask_sim = np.any([X_train_sim[:, idx] > np.percentile(X_train_sim[:, idx], percentile_threshold) for idx in feature_indices], axis=0)
normal_sim_data = X_train_sim[~anomaly_mask_sim]
anomaly_sim_data = X_train_sim[anomaly_mask_sim]
if len(anomaly_sim_data) == 0:
    anomaly_sim_data = X_train_sim[:int(0.03 * len(X_train_sim))]
logger.info(f"Normal data: {len(normal_sim_data)}, Anomalies selected: {len(anomaly_sim_data)}")

# Define early stopping for simulated data
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train AE on normal data only
logger.info("Starting AE training for simulated data on normal data only...")
ae_simulated.fit(normal_sim_data, normal_sim_data, validation_data=(X_val_sim, X_val_sim),
                 epochs=200, batch_size=32, callbacks=[early_stopping], verbose=0)
logger.info("AE training for simulated data completed.")

# Compute reconstruction errors with feature-weighted MSE and dynamic thresholding (FIXED)
reconstructions_simulated = ae_simulated.predict(data_scaled_simulated, verbose=0)
mse_simulated = np.mean(np.power(data_scaled_simulated - reconstructions_simulated, 2), axis=1)
# Feature-weighted MSE
feature_contributions = np.abs(data_scaled_simulated - reconstructions_simulated).mean(axis=0)
feature_importance = pd.Series(feature_contributions, index=feature_cols)
weights = feature_importance / feature_importance.sum()
weights[['CO_rolling_mean', 'NO2_rolling_mean', 'C6H6_rolling_mean']] *= 0.8  # Downweight rolling means
# Convert weights to NumPy array and reshape for broadcasting
weights_np = weights.to_numpy().reshape(1, -1)  # Shape: (1, 10)
weighted_mse = np.mean(np.power(data_scaled_simulated - reconstructions_simulated, 2) * weights_np, axis=1)
# Dynamic thresholding
window_size = 100
threshold_sim = np.percentile(weighted_mse, 96)  # Balanced threshold
rolling_threshold = pd.Series(weighted_mse).rolling(window=window_size, min_periods=1).quantile(0.96).fillna(threshold_sim)
df_simulated['Anomaly'] = (weighted_mse > rolling_threshold).astype(int)

# Ensemble with Isolation Forest
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.03, random_state=42)
iso_predictions = iso_forest.fit_predict(data_scaled_simulated)
iso_anomalies = (iso_predictions == -1).astype(int)
df_simulated['Anomaly'] = df_simulated['Anomaly'] & iso_anomalies

# Plot reconstruction error distribution
plt.figure(figsize=(10, 6))
sns.histplot(weighted_mse, bins=50, kde=True, label='Weighted MSE')
plt.axvline(threshold_sim, color='r', linestyle='--', label='Base Threshold (96th)')
plt.scatter(weighted_mse[anomaly_indices], np.zeros_like(weighted_mse[anomaly_indices]), color='red', label='True Anomalies', alpha=0.5)
plt.title('Weighted Reconstruction Error Distribution with Dynamic Threshold')
plt.xlabel('Weighted MSE')
plt.legend()
plt.savefig('mse_distribution_updated.png')
plt.close()

# Save predictions and model
df_simulated[['Date_Time'] + feature_cols + ['Anomaly']].to_csv('anomaly_predictions_simulated.csv', index=False)
ae_simulated.save('ae.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ## Evaluation
# Simulated ground truth
ground_truth_simulated = np.zeros(len(df))
ground_truth_simulated[anomaly_indices] = 1
print(f"Simulated Anomalies: {ground_truth_simulated.sum()} ({ground_truth_simulated.mean():.2%})")

# Evaluate simulated metrics
precision_simulated = precision_score(ground_truth_simulated, df_simulated['Anomaly'], zero_division=0)
recall_simulated = recall_score(ground_truth_simulated, df_simulated['Anomaly'], zero_division=0)
f1_simulated = f1_score(ground_truth_simulated, df_simulated['Anomaly'], zero_division=0)
print(f"Simulated - Precision: {precision_simulated:.2f}, Recall: {recall_simulated:.2f}, F1-Score: {f1_simulated:.2f}")

# Debug: Check predicted anomalies
print(f"Predicted Anomalies: {df_simulated['Anomaly'].sum()} ({df_simulated['Anomaly'].mean():.2%})")

# Save simulated confusion matrix
cm_simulated = confusion_matrix(ground_truth_simulated, df_simulated['Anomaly'])
plt.figure(figsize=(6, 4))
sns.heatmap(cm_simulated, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Simulated)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_simulated.png')
plt.close()

# Analyze feature importance for reconstruction errors
print("\nFeature Contributions to Reconstruction Errors (Simulated):")
print(feature_importance.sort_values(ascending=False))

# Plot reconstruction errors (simulated data)
plt.figure(figsize=(12, 6))
plt.plot(df_simulated['Date_Time'], weighted_mse, label='Weighted Reconstruction Error')
plt.plot(df_simulated['Date_Time'], rolling_threshold, color='r', linestyle='--', label='Rolling Threshold')
plt.scatter(df_simulated[df_simulated['Anomaly'] == 1]['Date_Time'], weighted_mse[df_simulated['Anomaly'] == 1], color='red', label='Anomaly')
plt.title('Weighted Reconstruction Errors and Anomalies (Simulated)')
plt.xlabel('Date_Time')
plt.ylabel('Weighted MSE')
plt.legend()
plt.savefig('reconstruction_errors_simulated.png')
plt.close()

# Debug: Compare reconstruction errors for normal vs. anomalous data
anomaly_mse = weighted_mse[anomaly_indices]
normal_mse = weighted_mse[~np.isin(np.arange(len(weighted_mse)), anomaly_indices)]
print(f"\nDebug - Mean Weighted MSE (Anomalies): {anomaly_mse.mean():.4f}, Mean Weighted MSE (Normal): {normal_mse.mean():.4f}")