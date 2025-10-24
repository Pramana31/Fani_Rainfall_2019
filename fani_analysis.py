"""
Tailored analysis script for "fani_cyclone_rainfall_2019.csv".
Save this file next to your data folder and run:
    python fani_analysis.py
Produces: EDA prints, feature engineering, XGBoost baseline, LSTM training (optional).
"""

import os
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "/mnt/data/fani_cyclone_rainfall_2019.csv"  # <- your uploaded file
OUT_DIR = "/mnt/data/fani_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load and inspect
df = pd.read_csv(DATA_PATH)
print("Loaded rows:", len(df))
print("Columns:", list(df.columns))
# rename expected cols for consistency
cols_map = {
    "Date": "time",
    "Latitude": "lat",
    "Longitude": "lon",
    "Rainfall_mm": "rain_mm",
    "WindSpeed_kmph": "wind_kmph",
    "Pressure_hPa": "pressure_hpa",
    "Humidity_%": "humidity",
    "Temperature_C": "temperature_c",
    "Distance_from_eye_km": "dist_eye_km",
    "Sea_Surface_Temp_C": "sst_c",
    "Precipitable_Water_mm": "pw_mm"
}
df = df.rename(columns=cols_map)
print("Columns after rename:", list(df.columns))

# 2. Parse time and basic cleaning
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
# Basic drop of rows without essential info
df = df.dropna(subset=['time','lat','lon','rain_mm']).reset_index(drop=True)
print("Rows after dropping missing essential values:", len(df))

# 3. Simple EDA prints and a quick plot saved to OUT_DIR
print(df[['time','lat','lon','rain_mm']].head(8).to_string(index=False))
print("\nRainfall stats:\n", df['rain_mm'].describe())

plt.figure(figsize=(10,4))
# aggregate total rainfall by day
daily = df.set_index('time').resample('D')['rain_mm'].sum().dropna()
daily.plot()
plt.title("Daily total rainfall (Fani dataset)")
plt.xlabel("Date"); plt.ylabel("Rainfall (mm)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "daily_rainfall.png"))
print("Saved daily rainfall plot to outputs.")

# 4. Feature engineering: lags, rolling sums, time features
df = df.sort_values(['lat','lon','time']).reset_index(drop=True)
# group by station via lat+lon grid (if station_id not present)
df['station_id'] = df['lat'].round(3).astype(str) + "_" + df['lon'].round(3).astype(str)

lags = [1,3,6,12,24]
for lag in lags:
    df[f"rain_lag_{lag}"] = df.groupby("station_id")["rain_mm"].shift(lag)

df["rain_24h"] = df.groupby("station_id")["rain_mm"].rolling(24, min_periods=1).sum().reset_index(0,drop=True)
df["hour"] = df["time"].dt.hour
df["doy"] = df["time"].dt.dayofyear

# 5. Target: predict next-hour rainfall
df["rain_tplus1"] = df.groupby("station_id")["rain_mm"].shift(-1)
# drop rows lacking target/feature history
feat_cols = ["dist_eye_km","wind_kmph","pressure_hpa","humidity","temperature_c","pw_mm"] + [f"rain_lag_{l}" for l in lags] + ["hour","doy"]
# keep only features that exist in df
feat_cols = [c for c in feat_cols if c in df.columns]
print("Using features:", feat_cols)
df_model = df.dropna(subset=feat_cols + ["rain_tplus1"]).copy()
print("Model rows after dropna:", len(df_model))

# 6. Train/test split by time (no leakage)
df_model = df_model.sort_values("time").reset_index(drop=True)
cut1 = df_model['time'].quantile(0.7)
cut2 = df_model['time'].quantile(0.85)
train = df_model[df_model['time']<=cut1]
val = df_model[(df_model['time']>cut1)&(df_model['time']<=cut2)]
test = df_model[df_model['time']>cut2]

X_train = train[feat_cols].values; y_train = train["rain_tplus1"].values
X_val = val[feat_cols].values; y_val = val["rain_tplus1"].values
X_test = test[feat_cols].values; y_test = test["rain_tplus1"].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# 7. Baseline persistence
if "rain_lag_1" in feat_cols:
    y_pred_persist = val["rain_lag_1"].values
    mae_persist = mean_absolute_error(y_val, y_pred_persist)
    rmse_persist = mean_squared_error(y_val, y_pred_persist, squared=False)
    print(f"Persistence baseline MAE: {mae_persist:.3f}, RMSE: {rmse_persist:.3f}")

# 8. Train XGBoost
dtrain = xgb.DMatrix(X_train_s, label=y_train)
dval = xgb.DMatrix(X_val_s, label=y_val)
params = {"objective":"reg:squarederror","eval_metric":"rmse","learning_rate":0.05,"max_depth":6,"subsample":0.8,"colsample_bytree":0.8,"seed":42}
bst = xgb.train(params, dtrain, num_boost_round=500, evals=[(dtrain,"train"),(dval,"val")], early_stopping_rounds=30, verbose_eval=50)
dtest = xgb.DMatrix(X_test_s)
y_pred_xgb = bst.predict(dtest)
print("XGBoost test MAE:", mean_absolute_error(y_test, y_pred_xgb), "RMSE:", mean_squared_error(y_test, y_pred_xgb, squared=False))

# 9. Save model and scaler
bst.save_model(os.path.join(OUT_DIR, "xgb_fani.model"))
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
print("Saved model and scaler to", OUT_DIR)

# 10. Diagnostics: feature importance and scatter plot
imp = bst.get_score(importance_type="gain")
imp_df = pd.DataFrame([(int(k.replace("f","")),v) for k,v in imp.items()], columns=["f","imp"]).sort_values("imp", ascending=False)
imp_df["feature"] = imp_df["f"].apply(lambda i: feat_cols[i])
print("Top features by gain:\n", imp_df[["feature","imp"]].head())

import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_xgb, s=6, alpha=0.5)
mx = max(y_test.max(), y_pred_xgb.max())
plt.plot([0,mx],[0,mx],"k--")
plt.xlabel("Observed rain_t+1 (mm)"); plt.ylabel("Predicted (mm)"); plt.title("Observed vs Predicted (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"obs_vs_pred_xgb.png"))
print("Saved obs_vs_pred_xgb.png")

# 11. Optional: quick LSTM nowcasting (only if you have tensorflow installed)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Masking
    LOOKBACK = 24
    def build_sequences(df_in, features, target, lookback=24):
        Xs, ys = [], []
        for sid, g in df_in.groupby("station_id"):
            g = g.sort_values("time").reset_index(drop=True)
            arr = g[features].values
            targ = g[target].values
            for i in range(len(g)-lookback):
                Xs.append(arr[i:i+lookback])
                ys.append(targ[i+lookback])
        return np.array(Xs), np.array(ys)
    X_seq_train, y_seq_train = build_sequences(train, feat_cols, "rain_tplus1", LOOKBACK)
    X_seq_val, y_seq_val = build_sequences(val, feat_cols, "rain_tplus1", LOOKBACK)
    X_seq_test, y_seq_test = build_sequences(test, feat_cols, "rain_tplus1", LOOKBACK)
    if len(X_seq_train)>0:
        # scale flattened
        n,t,f = X_seq_train.shape
        scaler_seq = StandardScaler().fit(X_seq_train.reshape(n,t*f))
        def scale_seq(X):
            n,t,f = X.shape
            Xf = X.reshape(n,t*f)
            Xfs = scaler_seq.transform(Xf)
            return Xfs.reshape(n,t,f)
        X_seq_train_s = scale_seq(X_seq_train); X_seq_val_s = scale_seq(X_seq_val); X_seq_test_s = scale_seq(X_seq_test)
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(LOOKBACK, len(feat_cols))))
        model.add(LSTM(64))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mae")
        model.fit(X_seq_train_s, y_seq_train, validation_data=(X_seq_val_s, y_seq_val), epochs=30, batch_size=128, verbose=2)
        y_pred_lstm = model.predict(X_seq_test_s).ravel()
        print("LSTM Test MAE:", mean_absolute_error(y_seq_test, y_pred_lstm))
        # save model
        model.save(os.path.join(OUT_DIR,"lstm_fani.h5"))
        joblib.dump(scaler_seq, os.path.join(OUT_DIR,"scaler_seq.joblib"))
    else:
        print("Not enough sequences to train LSTM. Skipping.")
except Exception as e:
    print("Skipping LSTM (tensorflow not available or error):", e)

print('All done. Check outputs in', OUT_DIR)
