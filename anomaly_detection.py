#!/usr/bin/env python
# coding: utf-8

# # IoT Anomaly Detection Project
#
# This notebook implements real-time anomaly detection for IoT sensor data using the UCI Air Quality dataset.
# The pipeline includes data analysis, preprocessing, Autoencoder (AE) training, evaluation with simulated ground truth, and visualization.
#
# **Dataset**: [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
# **Tools**: Pandas, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, NumPy, Pickle
# **Outputs**:
# - `preprocessed_data.csv`, `simulated_data.csv`, `anomaly_predictions_simulated.csv`
# - `correlation_matrix.png`, `feature_distributions.png`, `diurnal_patterns.png`, `confusion_matrix_simulated.png`
# - `ae.keras`, `scaler.pkl`

# ## Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
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
# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16', 'Hour'], errors='ignore')

# Replace -200 with NaN and impute
df[sensor_cols] = df[sensor_cols].replace(-200, np.nan)
df[sensor_cols] = df[sensor_cols].interpolate(method='linear', limit_direction='both')

# Compute derived features
df['CO_rolling_mean'] = df['CO(GT)'].rolling(window=3, min_periods=1).mean()
df['NO2_rolling_mean'] = df['NO2(GT)'].rolling(window=3, min_periods=1).mean()
df['C6H6_rolling_mean'] = df['C6H6(GT)'].rolling(window=3, min_periods=1).mean()
df['NOx_lag1'] = df['NOx(GT)'].shift(1).bfill()

# Add time-based features
df['Day_of_Week'] = df['Date_Time'].dt.dayofweek
df['Month'] = df['Date_Time'].dt.month
df['Is_Weekend'] = df['Date_Time'].dt.dayofweek.isin([5, 6]).astype(int)

# Define features for modeling
feature_cols = ['CO(GT)', 'NO2(GT)', 'NOx(GT)', 'C6H6(GT)', 'CO_rolling_mean', 'NO2_rolling_mean', 'C6H6_rolling_mean', 'NOx_lag1']
if not all(col in df.columns for col in feature_cols):
    raise ValueError("Missing feature columns in DataFrame")

# Normalize features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[feature_cols])

# Save preprocessed data
df.to_csv('preprocessed_data.csv', index=False)

# ## Autoencoder Training and Anomaly Detection
# Create simulated data with diverse anomalies (3%)
df_simulated = df.copy()
anomaly_indices = np.random.choice(len(df), size=int(0.03 * len(df)), replace=False)
for idx in anomaly_indices:
    r = np.random.random()
    if r < 0.2:  # Combined CO/NO2 Spike
        df_simulated.loc[idx, 'CO(GT)'] = np.random.uniform(7, 8)
        df_simulated.loc[idx, 'NO2(GT)'] = np.random.uniform(210, 230)
        df_simulated.loc[idx, 'NOx(GT)'] = np.random.uniform(550, 650)
    elif r < 0.4:  # Rapid CO Change + Moderate NO2
        df_simulated.loc[idx, 'NO2(GT)'] = np.random.uniform(140, 180)
        df_simulated.loc[idx, 'CO(GT)'] = np.random.uniform(4, 6)
    elif r < 0.6:  # Sustained High CO/NO2
        df_simulated.loc[idx, 'CO(GT)'] = np.random.uniform(7, 9)
        df_simulated.loc[idx, 'NO2(GT)'] = np.random.uniform(190, 210)
    elif r < 0.8:  # High Benzene + Normal CO
        df_simulated.loc[idx, 'C6H6(GT)'] = np.random.uniform(28, 35)
        df_simulated.loc[idx, 'CO(GT)'] = np.random.uniform(2, 4)
    else:  # Nighttime Rapid CO Change
        if df_simulated.loc[idx, 'Date_Time'].hour in [0, 1, 2, 3, 22, 23]:
            df_simulated.loc[idx, 'CO(GT)'] = np.random.uniform(3.5, 5)

# Compute derived features for simulated data
df_simulated['CO_rolling_mean'] = df_simulated['CO(GT)'].rolling(window=3, min_periods=1).mean()
df_simulated['NO2_rolling_mean'] = df_simulated['NO2(GT)'].rolling(window=3, min_periods=1).mean()
df_simulated['C6H6_rolling_mean'] = df_simulated['C6H6(GT)'].rolling(window=3, min_periods=1).mean()
df_simulated['NOx_lag1'] = df_simulated['NOx(GT)'].shift(1).bfill()

# Normalize simulated data
data_scaled_simulated = scaler.transform(df_simulated[feature_cols])
df_simulated.to_csv('simulated_data.csv', index=False)

"""
# Build AE for original data
input_dim = len(feature_cols)
latent_dim = 16
inputs = Input(shape=(input_dim,))

# Encoder
h = Dense(32, activation='relu')(inputs)
h = Dense(16, activation='relu')(h)
h = Dense(8, activation='relu')(h)
h = Dropout(0.2)(h)
latent = Dense(latent_dim, activation='relu')(h)

# Decoder
decoder_h = Dense(8, activation='relu')(latent)
decoder_h = Dense(16, activation='relu')(decoder_h)
decoder_h = Dense(32, activation='relu')(decoder_h)
decoder_h = Dropout(0.2)(decoder_h)
outputs = Dense(input_dim, activation='linear')(decoder_h)

# Define AE model
ae = Model(inputs, outputs)
ae.compile(optimizer='adam', loss='mse')

# Split data into training and validation sets
X_train, X_val, train_indices, val_indices = train_test_split(
    data_scaled, np.arange(len(data_scaled)), test_size=0.2, random_state=42
)
logger.info("Data split into training and validation sets.")

# Oversample anomalies for original data
key_features = ['NOx(GT)', 'CO(GT)', 'NO2_rolling_mean']
feature_indices = [feature_cols.index(feat) for feat in key_features]
percentile_threshold = 97
time_mask = df.iloc[train_indices]['Date_Time'].dt.hour.isin([0, 1, 2, 3, 22, 23]).values
anomaly_mask = np.any([X_train[:, idx] > np.percentile(X_train[:, idx], percentile_threshold) for idx in feature_indices], axis=0) & time_mask
normal_data = X_train[~anomaly_mask]
anomaly_data = X_train[anomaly_mask]
if len(anomaly_data) == 0:
    anomaly_data = X_train[:int(0.03 * len(X_train))]
oversampled_anomaly_data = np.repeat(anomaly_data, 3, axis=0)
X_train_resampled = np.vstack([normal_data, oversampled_anomaly_data])
np.random.shuffle(X_train_resampled)
logger.info(f"Oversampling completed using heuristic method for original data. Anomalies selected: {len(anomaly_data)}")

# Train AE on original data
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
logger.info("Starting AE training...")
ae.fit(X_train_resampled, X_train_resampled, validation_data=(X_val, X_val),
       epochs=200, batch_size=32, callbacks=[early_stopping], verbose=0)
logger.info("AE training completed.")

# Compute reconstruction errors with global thresholding for original data
reconstructions = ae.predict(data_scaled, verbose=0)
mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 97)
df['Anomaly'] = (mse > threshold).astype(int)
"""

# Split simulated data into training and validation sets
X_train_sim, X_val_sim, sim_train_indices, sim_val_indices = train_test_split(
    data_scaled_simulated, np.arange(len(data_scaled_simulated)), test_size=0.2, random_state=42
)
logger.info("Simulated data split into training and validation sets.")

# Define AE for simulated data
from tensorflow.keras.regularizers import l2
input_dim = len(feature_cols)
latent_dim = 16
inputs = Input(shape=(input_dim,))
h = Dense(32, activation='relu', kernel_regularizer=l2(0.03))(inputs)
h = Dense(16, activation='relu', kernel_regularizer=l2(0.03))(h)
h = Dense(8, activation='relu', kernel_regularizer=l2(0.03))(h)
h = Dropout(0.2)(h)
latent = Dense(latent_dim, activation='relu')(h)
decoder_h = Dense(8, activation='relu', kernel_regularizer=l2(0.03))(latent)
decoder_h = Dense(16, activation='relu', kernel_regularizer=l2(0.03))(decoder_h)
decoder_h = Dense(32, activation='relu', kernel_regularizer=l2(0.03))(decoder_h)
decoder_h = Dropout(0.2)(decoder_h)
outputs = Dense(input_dim, activation='linear')(decoder_h)
ae_simulated = Model(inputs, outputs)
ae_simulated.compile(optimizer='adam', loss='mse')
logger.info("AE model for simulated data created.")

key_features = ['NO2(GT)', 'C6H6(GT)']
feature_indices = [feature_cols.index(feat) for feat in key_features]
percentile_threshold = 96
time_mask_sim = df_simulated.iloc[sim_train_indices]['Date_Time'].dt.hour.isin([0, 1, 2, 3, 22, 23]).values
anomaly_mask_sim = np.any([X_train_sim[:, idx] > np.percentile(X_train_sim[:, idx], percentile_threshold) for idx in feature_indices], axis=0) & time_mask_sim
normal_sim_data = X_train_sim[~anomaly_mask_sim]
anomaly_sim_data = X_train_sim[anomaly_mask_sim]
if len(anomaly_sim_data) == 0:
    anomaly_sim_data = X_train_sim[:int(0.03 * len(X_train_sim))]
oversampled_anomaly_sim_data = np.repeat(anomaly_sim_data, 3, axis=0)
X_train_sim_resampled = np.vstack([normal_sim_data, oversampled_anomaly_sim_data])
np.random.shuffle(X_train_sim_resampled)
logger.info(f"Oversampling completed using heuristic method for simulated data. Anomalies selected: {len(anomaly_sim_data)}")

# Define early stopping for simulated data
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train AE on simulated data
logger.info("Starting AE training for simulated data...")
ae_simulated.fit(X_train_sim_resampled, X_train_sim_resampled, validation_data=(X_val_sim, X_val_sim),
                 epochs=200, batch_size=32, callbacks=[early_stopping], verbose=0)
logger.info("AE training for simulated data completed.")

# Compute reconstruction errors with global thresholding for simulated data
reconstructions_simulated = ae_simulated.predict(data_scaled_simulated, verbose=0)
mse_simulated = np.mean(np.power(data_scaled_simulated - reconstructions_simulated, 2), axis=1)
threshold_sim = np.percentile(mse_simulated, 95)
df_simulated['Anomaly'] = (mse_simulated > threshold_sim).astype(int)

# Save predictions and model
# df[['Date_Time'] + feature_cols + ['Anomaly']].to_csv('anomaly_predictions.csv', index=False)  # Commented out original data predictions
df_simulated[['Date_Time'] + feature_cols + ['Anomaly']].to_csv('anomaly_predictions_simulated.csv', index=False)
ae_simulated.save('ae.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ## Evaluation
"""
# Refined domain-based ground truth
co_threshold = np.percentile(df['CO_rolling_mean'].dropna(), 95)
nox_threshold = np.percentile(df['NOx(GT)'].dropna(), 95)
ground_truth_domain = (((df['CO_rolling_mean'] > co_threshold) |
                       (df['NOx(GT)'] > nox_threshold)) &
                      (df['Date_Time'].dt.hour.isin([0, 1, 2, 3, 22, 23]))).astype(int)
print(f"Domain-Based Anomalies: {ground_truth_domain.sum()} ({ground_truth_domain.mean():.2%})")
"""

# Simulated ground truth
ground_truth_simulated = np.zeros(len(df))
ground_truth_simulated[anomaly_indices] = 1
print(f"Simulated Anomalies: {ground_truth_simulated.sum()} ({ground_truth_simulated.mean():.2%})")

"""
# Evaluate domain-based metrics
precision_domain = precision_score(ground_truth_domain, df['Anomaly'])
recall_domain = recall_score(ground_truth_domain, df['Anomaly'])
f1_domain = f1_score(ground_truth_domain, df['Anomaly'])
print(f"\nDomain-Based - Precision: {precision_domain:.2f}, Recall: {recall_domain:.2f}, F1-Score: {f1_domain:.2f}")
"""

# Evaluate simulated metrics
precision_simulated = precision_score(ground_truth_simulated, df_simulated['Anomaly'])
recall_simulated = recall_score(ground_truth_simulated, df_simulated['Anomaly'])
f1_simulated = f1_score(ground_truth_simulated, df_simulated['Anomaly'])
print(f"Simulated - Precision: {precision_simulated:.2f}, Recall: {recall_simulated:.2f}, F1-Score: {f1_simulated:.2f}")

"""
# Save domain-based confusion matrix
cm_domain = confusion_matrix(ground_truth_domain, df['Anomaly'])
plt.figure(figsize=(6, 4))
sns.heatmap(cm_domain, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Domain-Based)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_domain.png')
plt.close()
"""

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
feature_contributions = np.abs(data_scaled_simulated - reconstructions_simulated).mean(axis=0)
feature_importance = pd.Series(feature_contributions, index=feature_cols)
print("\nFeature Contributions to Reconstruction Errors (Simulated):")
print(feature_importance.sort_values(ascending=False))

# Plot reconstruction errors (simulated data)
plt.figure(figsize=(12, 6))
plt.plot(df_simulated['Date_Time'], mse_simulated, label='Reconstruction Error')
plt.axhline(threshold_sim, color='r', linestyle='--', label='Threshold')
plt.scatter(df_simulated[df_simulated['Anomaly'] == 1]['Date_Time'], mse_simulated[df_simulated['Anomaly'] == 1], color='red', label='Anomaly')
plt.title('Reconstruction Errors and Anomalies (Simulated)')
plt.xlabel('Date_Time')
plt.ylabel('MSE')
plt.legend()
plt.savefig('reconstruction_errors_simulated.png')
plt.close()