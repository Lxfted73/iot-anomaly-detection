#!/usr/bin/env python

# coding: utf-8

# # IoT Anomaly Detection Project
# 
# This notebook implements real-time anomaly detection for IoT sensor data using the UCI Air Quality dataset. The pipeline includes data analysis, preprocessing, Variational Autoencoder (VAE) training, evaluation with domain-based and simulated ground truth, visualization, anomaly analysis, and simulation for Flask app testing.
# 
# **Dataset**: [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)  
# **Tools**: Pandas, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, NumPy, Requests, Flask, Pickle  
# **Outputs**:  
# - `preprocessed_ data.csv`, `simulated_data.csv`, `anomaly_predictions.csv`, `anomaly_analysis.csv`, `flask_simulation_data.csv`, `flask_simulation_results.csv`  
# - `correlation_matrix.png`, `feature_distributions.png`, `diurnal_patterns.png`, `co_anomaly_plot.png`, `no2_anomaly_plot.png`, `confusion_matrix_*.png`

# **Purpose**: Import necessary libraries for data analysis, VAE modeling, evaluation, visualization, and Flask app testing.  
# **Details**:  
# - `tensorflow.keras.backend` for VAE loss.  
# - `Lambda` for sampling layer.  
# - `EarlyStopping` for training optimization.  

# In[23]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Layer
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import pickle
from datetime import datetime, timedelta


# **Purpose**: Analyze the UCI Air Quality dataset to guide threshold selection and model training.  
# **Details**:  
# - Compute summary statistics, missing values, correlations, distributions, and diurnal patterns.  
# - Save plots as `correlation_matrix.png`, `feature_distributions.png`, `diurnal_patterns.png`.  

# In[24]:


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
plt.show()

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
plt.show()

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
plt.show()


# **Purpose**: Preprocess the dataset for VAE training.  
# **Details**:  
# - Drop unnecessary columns, impute missing values, compute derived features (`C6H6_rolling_mean`), and normalize.  
# - Save preprocessed data to `preprocessed_data.csv`.  

# In[26]:


# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16', 'Hour'])

# Replace -200 with NaN and impute
df[sensor_cols] = df[sensor_cols].replace(-200, np.nan)
df[sensor_cols] = df[sensor_cols].interpolate(method='linear', limit_direction='both')

# Compute derived features
df['CO_rolling_mean'] = df['CO(GT)'].rolling(window=3, min_periods=1).mean()
df['NO2_rolling_mean'] = df['NO2(GT)'].rolling(window=3, min_periods=1).mean()
df['C6H6_rolling_mean'] = df['C6H6(GT)'].rolling(window=3, min_periods=1).mean()
df.replace(-200, np.nan, inplace=True)
df['CO(GT)'] = df['CO(GT)'].interpolate(method='linear')
# Add Time-based Features
df['Day_of_Week'] = df['Date_Time'].dt.dayofweek
df['Month'] = df['Date_Time'].dt.month
df['Is_Weekend'] = df['Date_Time'].dt.dayofweek.isin([5, 6]).astype(int)

# Accentuate feature interactions
df['CO_NO2_interaction'] = df['CO(GT)'] * df['NO2(GT)']

# Normalize features
feature_cols = ['CO(GT)', 'NO2(GT)', 'NOx(GT)', 'CO_rolling_mean', 'NO2_rolling_mean', 'C6H6_rolling_mean']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[feature_cols])

# Save preprocessed data
df.to_csv('preprocessed_data.csv', index=False)


# **Purpose**: Train VAE models on original and simulated data to detect anomalies.  
# **Details**:  
# - Build a VAE with a custom loss layer (`VAELossLayer`) to compute reconstruction and KL-divergence losses.  
# - Train on mixed data (normal + 10x oversampled anomalies) with early stopping.  
# - Flag top 3% reconstruction errors as anomalies.  
# - Save to `anomaly_predictions.csv`, `vae.keras`, `scaler.pkl`.  

# In[8]:


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
        df_simulated.loc[idx, 'CO_diff'] = np.random.uniform(2, 3)
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
            df_simulated.loc[idx, 'CO_diff'] = np.random.uniform(1.8, 2.5)
            df_simulated.loc[idx, 'CO(GT)'] = np.random.uniform(3.5, 5)
df_simulated['CO_rolling_mean'] = df_simulated['CO(GT)'].rolling(window=3, min_periods=1).mean()
df_simulated['NO2_rolling_mean'] = df_simulated['NO2(GT)'].rolling(window=3, min_periods=1).mean()
df_simulated['C6H6_rolling_mean'] = df_simulated['C6H6(GT)'].rolling(window=3, min_periods=1).mean()
df_simulated['CO_diff'] = df_simulated['CO(GT)'].diff().fillna(0)
data_scaled_simulated = scaler.transform(df_simulated[feature_cols])
df_simulated.to_csv('simulated_data.csv', index=False)

# Define sampling function for VAE
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Custom VAE loss layer
class VAELossLayer(Layer):
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        x, x_decoded, z_mean, z_log_var = inputs
        # Reconstruction loss
        reconstruction_loss = K.mean(K.square(x - x_decoded), axis=-1)
        # KL-divergence loss
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # Total loss
        total_loss = K.mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)
        return x_decoded

# Build VAE
input_dim = len(feature_cols)  # 7 features
latent_dim = 4
inputs = Input(shape=(input_dim,))
h = Dense(16, activation='relu')(inputs)
h = Dropout(0.2)(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z = Lambda(sampling)([z_mean, z_log_var])
decoder_h = Dense(16, activation='relu')(z)
decoder_h = Dropout(0.2)(decoder_h)
x_decoded = Dense(input_dim)(decoder_h)

# Add VAE loss layer
outputs = VAELossLayer()([inputs, x_decoded, z_mean, z_log_var])
vae = Model(inputs, outputs)
vae.compile(optimizer='adam')

# Oversample anomalies
normal_data = data_scaled[df['Anomaly'] == 0] if 'Anomaly' in df else data_scaled
anomaly_data = data_scaled[df['Anomaly'] == 1] if 'Anomaly' in df else data_scaled[:int(0.03 * len(df))]
oversampled_anomaly_data = np.repeat(anomaly_data, 10, axis=0)
mixed_data = np.vstack([normal_data, oversampled_anomaly_data])
np.random.shuffle(mixed_data)

# Train VAE
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
vae.fit(mixed_data, epochs=200, batch_size=32, callbacks=[early_stopping], verbose=0)

# Compute reconstruction errors
reconstructions = vae.predict(data_scaled, verbose=0)
mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 97)
df['Anomaly'] = (mse > threshold).astype(int)

# Train VAE on simulated data
vae_simulated = Model(inputs, outputs)
vae_simulated.compile(optimizer='adam')
normal_sim_data = data_scaled_simulated[df_simulated['Anomaly'] == 0] if 'Anomaly' in df_simulated else data_scaled_simulated
anomaly_sim_data = data_scaled_simulated[df_simulated['Anomaly'] == 1] if 'Anomaly' in df_simulated else data_scaled_simulated[:int(0.03 * len(df))]
oversampled_anomaly_sim_data = np.repeat(anomaly_sim_data, 10, axis=0)
mixed_sim_data = np.vstack([normal_sim_data, oversampled_anomaly_sim_data])
np.random.shuffle(mixed_sim_data)
vae_simulated.fit(mixed_sim_data, epochs=200, batch_size=32, callbacks=[early_stopping], verbose=0)

# Compute reconstruction errors for simulated data
reconstructions_simulated = vae_simulated.predict(data_scaled_simulated, verbose=0)
mse_simulated = np.mean(np.power(data_scaled_simulated - reconstructions_simulated, 2), axis=1)
threshold_simulated = np.percentile(mse_simulated, 97)
df_simulated['Anomaly'] = (mse_simulated > threshold_simulated).astype(int)

# Save predictions and model
df[['Date_Time'] + feature_cols + ['Anomaly']].to_csv('anomaly_predictions.csv', index=False)
df_simulated[['Date_Time'] + feature_cols + ['Anomaly']].to_csv('anomaly_predictions_simulated.csv', index=False)
vae.save('vae.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# **Purpose**: Evaluate the VAE using refined domain-based, literal limit, and simulated ground truth.  
# **Details**:  
# - Use stricter thresholds for ~2–3% anomalies.  
# - Compute precision, recall, F1-score, and save confusion matrices.  

# In[9]:


# Refined domain-based ground truth
ground_truth_domain = (((df['CO_rolling_mean'] > 14) & (df['NO2(GT)'] > 280)) | 
                      (df['NOx(GT)'] > 900) | 
                      (df['C6H6(GT)'] > 35) | 
                      (df['NMHC(GT)'] > 650) | 
                      (df['CO_diff'].abs() > 2.5)).astype(int)
print(f"Domain-Based Anomalies: {ground_truth_domain.sum()} ({ground_truth_domain.mean():.2%})")

# Literal limit ground truth
ground_truth_literal = ((df['CO_rolling_mean'] > 8) | 
                       (df['NO2(GT)'] > 200) | 
                       (df['NOx(GT)'] > 400) | 
                       (df['C6H6(GT)'] > 5) | 
                       (df['NMHC(GT)'] > 200)).astype(int)
print(f"Literal Limit Anomalies: {ground_truth_literal.sum()} ({ground_truth_literal.mean():.2%})")

# Simulated ground truth
ground_truth_simulated = np.zeros(len(df))
ground_truth_simulated[anomaly_indices] = 1
print(f"Simulated Anomalies: {ground_truth_simulated.sum()} ({ground_truth_simulated.mean():.2%})")

# Evaluate metrics
precision_domain = precision_score(ground_truth_domain, df['Anomaly'])
recall_domain = recall_score(ground_truth_domain, df['Anomaly'])
f1_domain = f1_score(ground_truth_domain, df['Anomaly'])
print(f"\nDomain-Based - Precision: {precision_domain:.2f}, Recall: {recall_domain:.2f}, F1-Score: {f1_domain:.2f}")

precision_literal = precision_score(ground_truth_literal, df['Anomaly'])
recall_literal = recall_score(ground_truth_literal, df['Anomaly'])
f1_literal = f1_score(ground_truth_literal, df['Anomaly'])
print(f"Literal Limit - Precision: {precision_literal:.2f}, Recall: {recall_literal:.2f}, F1-Score: {f1_literal:.2f}")

precision_simulated = precision_score(ground_truth_simulated, df_simulated['Anomaly'])
recall_simulated = recall_score(ground_truth_simulated, df_simulated['Anomaly'])
f1_simulated = f1_score(ground_truth_simulated, df_simulated['Anomaly'])
print(f"Simulated - Precision: {precision_simulated:.2f}, Recall: {recall_simulated:.2f}, F1-Score: {f1_simulated:.2f}")

# Save confusion matrices
cm_domain = confusion_matrix(ground_truth_domain, df['Anomaly'])
plt.figure(figsize=(6, 4))
sns.heatmap(cm_domain, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Domain-Based)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_domain.png')
plt.show()

cm_literal = confusion_matrix(ground_truth_literal, df['Anomaly'])
plt.figure(figsize=(6, 4))
sns.heatmap(cm_literal, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Literal Limit)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_literal.png')
plt.show()

cm_simulated = confusion_matrix(ground_truth_simulated, df_simulated['Anomaly'])
plt.figure(figsize=(6, 4))
sns.heatmap(cm_simulated, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Simulated)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_simulated.png')
plt.show()


# In[ ]:





# In[ ]:




