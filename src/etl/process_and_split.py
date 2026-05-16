import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

def process_and_split():
    # Load config
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "config.json"))
    with open(config_path, "r") as f:
        config = json.load(f)
        
    print("Loading raw data...")
    raw_path = config["dataset"]["raw_data_path"]
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found. Please download dataset.")
        return

    # Check if running within an experiment
    exp_dir = os.environ.get("EXPERIMENT_DIR", "")
    base_data_dir = os.path.join(exp_dir, "data") if exp_dir else "data"
    
    df = pd.read_csv(raw_path)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    
    print("Processing data (ETL)...")
    time_col = config["dataset"]["time_column"]
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)
    df.sort_index(inplace=True)
    
    # Resample
    resample_freq = config["dataset"]["resample_freq"]
    df_resampled = df.resample(resample_freq).mean(numeric_only=True).ffill().dropna()
    
    # -------------------------------------------------------------------
    # Feature Engineering Phase
    # -------------------------------------------------------------------
    feat_config = config.get("feature_engineering", {})
    
    # 1. Add Time Features
    if feat_config.get("add_time_features", False):
        print("Adding Time Features (Hour, DayOfWeek, Month)...")
        df_resampled['hour'] = df_resampled.index.hour
        df_resampled['dayofweek'] = df_resampled.index.dayofweek
        df_resampled['month'] = df_resampled.index.month
        # Add Sin/Cos transforms for cyclic nature of time
        df_resampled['hour_sin'] = np.sin(2 * np.pi * df_resampled['hour'] / 24)
        df_resampled['hour_cos'] = np.cos(2 * np.pi * df_resampled['hour'] / 24)
        
    # 2. Add Lag Features
    lag_config = feat_config.get("lag_features", {})
    lag_cols = lag_config.get("columns", [])
    lags = lag_config.get("lags", [])
    if lag_cols and lags:
        print(f"Adding Lag Features for {lag_cols} at lags {lags}...")
        for col in lag_cols:
            if col in df_resampled.columns:
                for lag in lags:
                    df_resampled[f'{col}_lag_{lag}'] = df_resampled[col].shift(lag)
                    
    # Drop rows with NaNs caused by lagging
    if lag_cols and lags:
        df_resampled.dropna(inplace=True)
    # -------------------------------------------------------------------
    
    # Create target
    source_col = config["forecasting"]["target_source_column"]
    shift_steps = config["forecasting"]["target_shift_steps"]
    target_col = config["forecasting"]["target_column_name"]
    
    df_resampled[target_col] = df_resampled[source_col].shift(shift_steps)
    df_resampled.dropna(inplace=True)
    
    processed_dir = os.path.join(base_data_dir, "2_etl")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "processed.csv")
    df_resampled.to_csv(processed_path)
    print(f"Processed data saved to {processed_path}. Shape: {df_resampled.shape}")

    # Plot Distribution
    plot_distribution(df_resampled, processed_dir)

    print("Splitting into Train, Val, Test...")
    test_size_ratio = config["validation"]["global_test_size_ratio"]
    test_size = int(len(df_resampled) * test_size_ratio)
    train_val_df = df_resampled.iloc[:-test_size]
    test_df = df_resampled.iloc[-test_size:]
    
    splits_dir = os.path.join(base_data_dir, "3_kfold_splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    # Save Global Test Set
    test_df.to_csv(os.path.join(splits_dir, "test.csv"))
    
    # Meta data to track N counts and structure
    split_meta = {
        "dataset_total_n": len(df_resampled),
        "global_test_n": len(test_df),
        "folds": {}
    }

    # ตั้งค่าตัวแปรเพื่อควบคุมขนาด N (จำนวนบรรทัด) ภายใน Fold อย่างละเอียด
    n_splits = config["validation"]["n_splits"]
    # test_size_in_fold: จำนวน N สำหรับชุด Validation ของทุกๆ Fold
    # max_train_size: จำกัดให้ Train มี N มากสุดเท่าใด (ถ้าอยากให้ Train มีขนาดเท่ากันทุก Fold) 
    # (สามารถใส่ max_train_size=None ถ้าอยากให้ Train ขยายขนาดใหญ่ขึ้นเรื่อยๆ ตามลำดับเวลา)
    test_size_in_fold = int(len(train_val_df) / (n_splits + 1)) 
    
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size_in_fold, max_train_size=None)
    
    fold = 1
    for train_index, val_index in tscv.split(train_val_df):
        train_data = train_val_df.iloc[train_index]
        val_data = train_val_df.iloc[val_index]
        
        fold_dir = os.path.join(splits_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        train_data.to_csv(os.path.join(fold_dir, "train.csv"))
        val_data.to_csv(os.path.join(fold_dir, "val.csv"))
        
        split_meta["folds"][f"fold_{fold}"] = {
            "train_n": len(train_data),
            "val_n": len(val_data)
        }
        fold += 1
        
    with open(os.path.join(splits_dir, "split_meta.json"), "w") as f:
        json.dump(split_meta, f, indent=4)
        
    print("ETL and Splitting complete!")

def plot_distribution(df, save_dir, target_col='target'):
    import seaborn as sns
    # Set seaborn style for better aesthetics
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution Plot
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=target_col, kde=True, color='skyblue', edgecolor='black', bins=50)
    plt.title(f"Target ({target_col}) Distribution (N={len(df)})")
    plt.xlabel(f"{target_col} Value")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, y=target_col, color='salmon')
    plt.title(f"Target ({target_col}) Boxplot (Outliers)")
    plt.ylabel(f"{target_col} Value")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "data_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Time Series Plot (Raw Overview)
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, df[target_col], color='teal', linewidth=1)
    plt.title(f"Target ({target_col}) Over Time (Processed Raw)")
    plt.xlabel("Time")
    plt.ylabel(f"{target_col} Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "raw_timeseries.png"), dpi=300)
    plt.close()
    
    print(f"Saved distribution and timeseries plots to {save_dir}")

if __name__ == "__main__":
    process_and_split()
