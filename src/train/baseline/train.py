import os
import sys
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.models.baseline_model import BaselineModel
from src.utils.data_utils import prepare_tabular_data

def train_baseline():
    exp_dir = os.environ.get("EXPERIMENT_DIR", "")
    base_data_dir = os.path.join(exp_dir, "data") if exp_dir else "data"
    splits_dir = os.path.join(base_data_dir, "3_kfold_splits")
    
    base_results_dir = exp_dir if exp_dir else "results"
    results_dir = os.path.join(base_results_dir, "baseline")
    os.makedirs(results_dir, exist_ok=True)
    
    folds = sorted([f for f in os.listdir(splits_dir) if f.startswith('fold_')])
    
    # Baseline picks these specific features
    feature_cols = ['TP2', 'TP3', 'Motor_current', 'Oil_temperature']
    target_col = 'target'
    
    val_metrics = {}
    test_metrics = {}
    
    # Load Global Test Set
    global_test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    X_global_test, y_global_test = prepare_tabular_data(global_test_df, feature_cols, target_col)
    
    for fold in folds:
        print(f"--- Training Baseline on {fold} ---")
        train_df = pd.read_csv(os.path.join(splits_dir, fold, "train.csv"))
        val_df = pd.read_csv(os.path.join(splits_dir, fold, "val.csv"))
        
        X_train, y_train = prepare_tabular_data(train_df, feature_cols, target_col)
        X_val, y_val = prepare_tabular_data(val_df, feature_cols, target_col)
        
        model = BaselineModel()
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
        test_metrics[fold] = {"MSE": test_mse, "MAE": test_mae}
        
        # Save test predictions for plotting (overwrites so it keeps the last fold)
        np.save(os.path.join(results_dir, "last_test_preds.npy"), test_preds)
        
        # Save model
        joblib.dump(model.model, os.path.join(results_dir, f"{fold}_model.pkl"))
        
    # We save test metrics as the primary metrics for the global benchmark
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    print("Baseline training complete!")

if __name__ == "__main__":
    train_baseline()
