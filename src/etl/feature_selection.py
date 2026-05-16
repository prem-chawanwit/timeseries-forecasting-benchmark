import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.data_utils import prepare_tabular_data

def run_feature_selection():
    print("Running Feature Selection Phase...")
    
    exp_dir = os.environ.get("EXPERIMENT_DIR", "")
    base_data_dir = os.path.join(exp_dir, "data") if exp_dir else "data"
    splits_dir = os.path.join(base_data_dir, "3_kfold_splits")
    
    # We will use Fold 1 to determine feature importance (to avoid data leakage from Test set)
    train_path = os.path.join(splits_dir, "fold_1", "train.csv")
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return
        
    # Load config
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "config.json"))
    with open(config_path, "r") as f:
        config = json.load(f)
        
    df_train = pd.read_csv(train_path)
    
    # Exclude timestamp and specific features (to prevent naive shift/copying)
    target_col = config["forecasting"]["target_column_name"]
    time_col = config["dataset"]["time_column"]
    exclude_list = config["forecasting"]["exclude_features"] + [time_col, target_col]
    
    all_features = [col for col in df_train.columns if col not in exclude_list]
    
    X_train, y_train = prepare_tabular_data(df_train, all_features, target_col)
    
    # Train a quick XGBoost model to get Feature Importances
    model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    importances = model.feature_importances_
    
    # Sort features
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [all_features[i] for i in sorted_idx]
    sorted_importances = [float(importances[i]) for i in sorted_idx]
    
    # Select Top K Features automatically
    top_k = config["forecasting"]["top_k_features"]
    selected_features = sorted_features[:top_k]
    print(f"Selected Top {top_k} Features: {selected_features}")
    
    # Save the selected features to a JSON file so train scripts can use them
    feature_meta = {
        "all_features_ranked": sorted_features,
        "importances": sorted_importances,
        "selected_features": selected_features
    }
    
    # Save to the ETL directory
    etl_dir = os.path.join(base_data_dir, "2_etl")
    os.makedirs(etl_dir, exist_ok=True)
    with open(os.path.join(etl_dir, "selected_features.json"), "w") as f:
        json.dump(feature_meta, f, indent=4)
        
    # Plot 1: Feature Importance Ranking
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="magma")
    plt.title("Feature Importance Ranking (from Fold 1)")
    plt.xlabel("XGBoost Information Gain")
    plt.tight_layout()
    plt.savefig(os.path.join(etl_dir, "feature_ranking.png"), dpi=300)
    plt.close()
    
    # Plot 2: Selected Features vs Target over Time
    # To avoid plotting too much data, we can take a sample or plot the entire Fold 1 train set
    plot_df = df_train.copy()
    if 'timestamp' in plot_df.columns:
        plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'])
        plot_df.set_index('timestamp', inplace=True)
        
    n_plots = len(selected_features) + 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 3 * n_plots), sharex=True)
    
    # Plot Target
    axes[0].plot(plot_df.index, plot_df[target_col], color='black', label=f'Target ({target_col})')
    axes[0].set_title(f"Target Variable: {target_col}")
    axes[0].legend(loc="upper right")
    
    # Plot Selected Features
    colors = sns.color_palette("husl", len(selected_features))
    for i, (feature, color) in enumerate(zip(selected_features, colors), 1):
        axes[i].plot(plot_df.index, plot_df[feature], color=color, label=f'Feature: {feature}')
        axes[i].set_title(f"Selected Feature #{i}: {feature}")
        axes[i].legend(loc="upper right")
        
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(os.path.join(etl_dir, "selected_features_timeseries.png"), dpi=300)
    plt.close()
    
    print("Feature Selection Phase Complete!")

if __name__ == "__main__":
    run_feature_selection()
