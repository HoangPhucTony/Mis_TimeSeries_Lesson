import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. LOGGING
def log_path(model_name, base_dir):
    log_dir = os.path.join(base_dir, model_name)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{model_name.lower()}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(model_name)

    logger.info("=" * 60)
    logger.info(f"MODEL: {model_name}")
    logger.info("=" * 60)

    return logger, log_file


# 2. SAVE PLOT
def save_plot(fig_name, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    if not fig_name.lower().endswith(".png"):
        fig_name += ".png"

    path = os.path.join(plot_dir, fig_name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return path

# 3. LOAD DATASET
def load_dataset(path):
    df = pd.read_csv(path)

    df["Datetime"] = pd.to_datetime(
        df[["YEAR", "Month", "Day", "Hour"]]
    )

    df = df.set_index("Datetime")
    df = df.sort_index()

    df = df.rename(columns={
        "Electric Load (MW)": "load_MW",
        "specific humidity": "humidity",
        "wind speed": "wind_speed"
    })

    return df

# 4. TIME SPLIT (KHÔNG SHUFFLE)
def time_split(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    return train_df, val_df, test_df


# 5. SCALE FEATURES
def scale_features(train_df, val_df, test_df):
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_df)
    val_scaled   = scaler.transform(val_df)
    test_scaled  = scaler.transform(test_df)
 # StandardScaler trả về numpy array, cần giữ timestamp index 
    train_scaled = pd.DataFrame(train_scaled, index=train_df.index, columns=train_df.columns)
    val_scaled   = pd.DataFrame(val_scaled,   index=val_df.index,   columns=val_df.columns)
    test_scaled  = pd.DataFrame(test_scaled,  index=test_df.index,  columns=test_df.columns)

    return train_scaled, val_scaled, test_scaled, scaler
