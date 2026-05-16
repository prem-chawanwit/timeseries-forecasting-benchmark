import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def analyze_time_series():
    # Load config
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    raw_path = config["dataset"]["raw_data_path"]
    time_col = config["dataset"]["time_column"]
    freq = config["dataset"]["resample_freq"]
    target_col = config["forecasting"]["target_source_column"]
    
    print(f"Loading data from {raw_path}...")
    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        print("Data file not found. Please check raw_data_path in config.json")
        return
        
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)
    df.sort_index(inplace=True)
    
    # Resample to align data perfectly
    df = df.resample(freq).mean(numeric_only=True).ffill().dropna()
    
    import datetime
    import shutil
    
    # Create timestamped EDA run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("eda_reports", f"eda_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save a copy of the config used
    shutil.copy(config_path, os.path.join(run_dir, "config_used.json"))
    
    print("\n" + "="*50)
    print(f"📊 TIME SERIES EDA (Run: eda_{timestamp})")
    print("="*50)
    
    # 1. Target Autocorrelation (To find best self lags)
    lags = list(range(1, 49)) # Test up to 48 steps
    autocorrs = [df[target_col].corr(df[target_col].shift(lag)) for lag in lags]
    
    plt.figure(figsize=(14, 5))
    plt.bar(lags, autocorrs, color='teal')
    plt.axhline(0, color='black', lw=1)
    plt.title(f"Autocorrelation of {target_col}")
    plt.xlabel(f"Lag (steps of {freq})")
    plt.ylabel("Correlation")
    plt.xticks(np.arange(0, 49, 2))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "autocorrelation.png"), dpi=300)
    plt.close()
    
    top_lags_idx = np.argsort(np.abs(autocorrs))[::-1][:5]
    top_lags = [lags[i] for i in top_lags_idx]
    
    report_lines = []
    report_lines.append(f"EDA Run: eda_{timestamp}\n")
    report_lines.append(f"1. Recommended Self-Lags for [{target_col}]:")
    report_lines.append(f"Top 5 Lags with highest correlation: {sorted(top_lags)}\n")
    
    # 2. Cross-correlation with other features
    exclude = config["forecasting"].get("exclude_features", [])
    features = [c for c in df.columns if c not in exclude and c != target_col]
    
    report_lines.append(f"2. Cross-Correlation Analysis (0 to 48 lags):")
    
    # Plot feature cross-correlations
    plt.figure(figsize=(14, 8))
    
    for feat in features:
        cross_corrs = [df[target_col].corr(df[feat].shift(lag)) for lag in range(0, 49)]
        best_lag_idx = np.argmax(np.abs(cross_corrs))
        best_corr = cross_corrs[best_lag_idx]
        
        # Plot the CCF curve for this feature
        plt.plot(range(0, 49), cross_corrs, marker='o', markersize=3, label=f"{feat} (Best: Lag {best_lag_idx})")
        
        if abs(best_corr) > 0.3:
            report_lines.append(f" - [{feat}]: Best correlation at Lag {best_lag_idx} (Corr = {best_corr:.2f})")
            
    plt.axhline(0, color='black', lw=1, linestyle='--')
    plt.title(f"Cross-Correlation: Features vs {target_col}")
    plt.xlabel(f"Lag (steps of {freq})")
    plt.ylabel("Correlation")
    plt.xticks(np.arange(0, 49, 2))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "feature_cross_correlation.png"), dpi=300)
    plt.close()
    
    # Write report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    with open(os.path.join(run_dir, "lag_recommendations.txt"), "w") as f:
        f.write(report_text)
        
    print(f"\n✅ วิเคราะห์เสร็จสิ้น! บันทึกข้อมูลและกราฟทั้งหมดไว้ที่: {run_dir}")

if __name__ == "__main__":
    analyze_time_series()
