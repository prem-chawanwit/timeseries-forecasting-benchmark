import os
import sys
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.models.lstm_model import LSTMForecaster
from src.utils.data_utils import create_sequences

def train_lstm():
    exp_dir = os.environ.get("EXPERIMENT_DIR", "")
    base_data_dir = os.path.join(exp_dir, "data") if exp_dir else "data"
    splits_dir = os.path.join(base_data_dir, "3_kfold_splits")
    
    base_results_dir = exp_dir if exp_dir else "results"
    results_dir = os.path.join(base_results_dir, "lstm")
    os.makedirs(results_dir, exist_ok=True)
    
    folds = sorted([f for f in os.listdir(splits_dir) if f.startswith('fold_')])
    
    # LSTM picks features
    feature_cols = ['TP2', 'TP3', 'Motor_current', 'Oil_temperature', 'DV_pressure']
    target_col = 'target'
    seq_length = 5
    epochs = 20
    
    val_metrics = {}
    test_metrics = {}
    
    global_test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    
    for fold in folds:
        print(f"--- Training LSTM on {fold} ---")
        train_df = pd.read_csv(os.path.join(splits_dir, fold, "train.csv"))
        val_df = pd.read_csv(os.path.join(splits_dir, fold, "val.csv"))
        
        # Scale features
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df[feature_cols])
        val_scaled = scaler.transform(val_df[feature_cols])
        test_scaled = scaler.transform(global_test_df[feature_cols])
        
        train_target = train_df[target_col].values
        val_target = val_df[target_col].values
        test_target = global_test_df[target_col].values
        
        X_train, y_train = create_sequences(train_scaled, train_target, seq_length)
        X_val, y_val = create_sequences(val_scaled, val_target, seq_length)
        X_test, y_test = create_sequences(test_scaled, test_target, seq_length)
        
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val)
        X_test_t = torch.FloatTensor(X_test)
        
        model = LSTMForecaster(input_size=len(feature_cols))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Training
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
        # Validation Eval
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).numpy().flatten()
            test_preds = model(X_test_t).numpy().flatten()
            
        val_mse = mean_squared_error(y_val, val_preds)
        val_mae = mean_absolute_error(y_val, val_preds)
        val_metrics[fold] = {"MSE": float(val_mse), "MAE": float(val_mae)}
        print(f"[{fold}] Validation MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")
        
        # Test Eval
        test_mse = mean_squared_error(y_test, test_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_metrics[fold] = {"MSE": float(test_mse), "MAE": float(test_mae)}
        
        # Save test predictions
        np.save(os.path.join(results_dir, "last_test_preds.npy"), test_preds)
        
        torch.save(model.state_dict(), os.path.join(results_dir, f"{fold}_model.pth"))
        
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    print("LSTM training complete!")

if __name__ == "__main__":
    train_lstm()
