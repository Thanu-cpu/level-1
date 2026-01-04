import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
df = pd.read_csv('anomaly_detection.csv')
features = ['login_duration_min', 'data_accessed_MB', 'files_downloaded']

# 2. Preprocessing & Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# 3. Statistical Method: Interquartile Range (IQR)
def flag_iqr(data, col):
    Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
    IQR = Q3 - Q1
    return (data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))

df['iqr_anomaly'] = False
for col in features:
    df['iqr_anomaly'] |= flag_iqr(df, col)

# 4. Unsupervised ML: Isolation Forest
# Contamination set to 5% based on expected breach scale
model = IsolationForest(contamination=0.05, random_state=42)
df['ml_anomaly'] = model.fit_predict(X_scaled)
df['ml_anomaly'] = df['ml_anomaly'].map({1: False, -1: True})

# 5. Suspiciousness Scoring (Distance from Mean)
z_scores = (df[features] - df[features].mean()) / df[features].std()
df['risk_score'] = z_scores.abs().sum(axis=1)

# 6. Reporting Top 5 Suspects
suspects = df.sort_values(by='risk_score', ascending=False).head(5)
print("TOP 5 FORENSIC SUSPECTS:")
print(suspects[['user_id', 'remote_access', 'risk_score']])

# 7. Save Results
df.to_csv('forensic_report_results.csv', index=False)
