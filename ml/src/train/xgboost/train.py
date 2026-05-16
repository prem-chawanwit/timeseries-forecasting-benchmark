import os
import sys
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.models.xgboost_model import XGBoostModel
from src.utils.data_utils import prepare_tabular_data

def train_xgboost():
    exp_dir = os.environ.get("EXPERIMENT_DIR", "")
    base_data_dir = os.path.join(exp_dir, "data") if exp_dir else "data"
    splits_dir = os.path.join(base_data_dir, "3_kfold_splits")
    
    base_results_dir = exp_dir if exp_dir else "results"
    results_dir = os.path.join(base_results_dir, "xgboost")
    os.makedirs(results_dir, exist_ok=True)
    
    folds = sorted([f for f in os.listdir(splits_dir) if f.startswith('fold_')])
    
    # Load features
    feature_cols = ['TP2', 'TP3', 'Motor_current', 'Oil_temperature', 'DV_pressure', 'Oil_level']
    
    # Try reading from ETL selection first (Dynamic Selection Phase)
    etl_features_path = os.path.join(base_data_dir, "2_etl", "selected_features.json")
    if os.path.exists(etl_features_path):
        with open(etl_features_path, "r") as f:
            feature_meta = json.load(f)
            feature_cols = feature_meta["selected_features"]
    elif "FEATURE_COLS" in os.environ:
        feature_cols = os.environ["FEATURE_COLS"].split(",")
    # Load config
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "config.json"))
    with open(config_path, "r") as f:
        config = json.load(f)
        
    target_col = config["forecasting"]["target_column_name"]
    xgb_params = config["model_params"]["xgboost"]
    
    val_metrics = {}
    test_metrics = {}
    
    # Load Global Test Set
    global_test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    X_global_test, y_global_test = prepare_tabular_data(global_test_df, feature_cols, target_col)
    
    for fold in folds:
        print(f"--- Training XGBoost on {fold} ---")
        train_df = pd.read_csv(os.path.join(splits_dir, fold, "train.csv"))
        val_df = pd.read_csv(os.path.join(splits_dir, fold, "val.csv"))
        
        X_train, y_train = prepare_tabular_data(train_df, feature_cols, target_col)
        X_val, y_val = prepare_tabular_data(val_df, feature_cols, target_col)
        
        model = XGBoostModel(**xgb_params)
        model.fit(X_train, y_train)
        
        # Validation Eval
        val_preds = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_preds)
        val_mae = mean_absolute_error(y_val, val_preds)
        val_metrics[fold] = {"MSE": val_mse, "MAE": val_mae}
        print(f"[{fold}] Validation MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")
        
        # Global Test Eval
        test_preds = model.predict(X_global_test)
        test_mse = mean_squared_error(y_global_test, test_preds)
        test_mae = mean_absolute_error(y_global_test, test_preds)
        test_metrics[fold] = {"MSE": float(test_mse), "MAE": float(test_mae)}
        
        # Save test predictions
        np.save(os.path.join(results_dir, "last_test_preds.npy"), test_preds)
        
        # Save model
        model.model.save_model(os.path.join(results_dir, f"{fold}_model.json"))
        
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    print("XGBoost training complete!")

if __name__ == "__main__":
    train_xgboost()
