import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

def process_and_split():
    print("Loading raw data...")
    raw_path = "data/1_raw/metropt-3-dataset/MetroPT3(AirCompressor).csv"
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found. Please run download_dataset.py first.")
        return

    # Check if running within an experiment
    exp_dir = os.environ.get("EXPERIMENT_DIR", "")
    base_data_dir = os.path.join(exp_dir, "data") if exp_dir else "data"
    
    df = pd.read_csv(raw_path)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    
    print("Processing data (ETL)...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Resample to 1H
    df_resampled = df.resample('1h').mean(numeric_only=True).ffill().dropna()
    
    # Create target
    df_resampled['target'] = df_resampled['TP2'].shift(-1)
    df_resampled.dropna(inplace=True)
    
    processed_dir = os.path.join(base_data_dir, "2_etl")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "processed.csv")
    df_resampled.to_csv(processed_path)
    print(f"Processed data saved to {processed_path}. Shape: {df_resampled.shape}")

    # Plot Distribution
    plot_distribution(df_resampled, processed_dir)

    print("Splitting into Train, Val, Test...")
    test_size = int(len(df_resampled) * 0.15)
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
    n_splits = 10
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

def plot_distribution(df, save_dir):
    import seaborn as sns
    # Set seaborn style for better aesthetics
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution Plot
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='target', kde=True, color='skyblue', edgecolor='black', bins=50)
    plt.title(f"Target (TP2) Distribution (N={len(df)})")
    plt.xlabel("TP2 Value")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, y='target', color='salmon')
    plt.title("Target (TP2) Boxplot (Outliers)")
    plt.ylabel("TP2 Value")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "data_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Time Series Plot (Raw Overview)
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, df['target'], color='teal', linewidth=1)
    plt.title("Target (TP2) Over Time (Processed Raw)")
    plt.xlabel("Time")
    plt.ylabel("TP2 Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "raw_timeseries.png"), dpi=300)
    plt.close()
    
    print(f"Saved distribution and timeseries plots to {save_dir}")

if __name__ == "__main__":
    process_and_split()
