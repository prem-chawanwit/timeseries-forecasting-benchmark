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
    
    # Load features
    feature_cols = ['TP2', 'TP3', 'Motor_current', 'Oil_temperature', 'DV_pressure']
    
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
    seq_length = config["model_params"]["lstm"]["seq_length"]
    epochs = config["model_params"]["lstm"]["epochs"]
    learning_rate = config["model_params"]["lstm"].get("learning_rate", 0.01)
    
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
        
        # Scale targets
        target_scaler = StandardScaler()
        train_target = target_scaler.fit_transform(train_df[[target_col]]).flatten()
        val_target = target_scaler.transform(val_df[[target_col]]).flatten()
        test_target = target_scaler.transform(global_test_df[[target_col]]).flatten()
        
        X_train, y_train_s = create_sequences(train_scaled, train_target, seq_length)
        X_val, y_val_s = create_sequences(val_scaled, val_target, seq_length)
        X_test, y_test_s = create_sequences(test_scaled, test_target, seq_length)
        
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train_s).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val)
        X_test_t = torch.FloatTensor(X_test)
        
        model = LSTMForecaster(input_size=len(feature_cols))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training with Early Stopping
        patience = 10
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        
        train_losses = []
        val_losses = []
        
        y_val_t = torch.FloatTensor(y_val_s).unsqueeze(1)
        
        for epoch in range(epochs):
            # Train step
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                
            train_losses.append(loss.item())
            val_losses.append(val_loss)
            
            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                import copy
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print(f"[{fold}] Early stopping triggered at epoch {epoch+1}")
                break
                
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        # Plot Learning Curve
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss (MSE)')
        plt.plot(val_losses, label='Validation Loss (MSE)')
        plt.title(f'LSTM Learning Curve - {fold}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"learning_curve_{fold}.png"), dpi=300)
        plt.close()
            
        # Validation Eval
        model.eval()
        with torch.no_grad():
            val_preds_s = model(X_val_t).numpy().flatten()
            test_preds_s = model(X_test_t).numpy().flatten()
            
        # Inverse transform to compute real-world metrics
        val_preds = target_scaler.inverse_transform(val_preds_s.reshape(-1, 1)).flatten()
        y_val_real = target_scaler.inverse_transform(y_val_s.reshape(-1, 1)).flatten()
        
        test_preds = target_scaler.inverse_transform(test_preds_s.reshape(-1, 1)).flatten()
        y_test_real = target_scaler.inverse_transform(y_test_s.reshape(-1, 1)).flatten()
            
        val_mse = mean_squared_error(y_val_real, val_preds)
        val_mae = mean_absolute_error(y_val_real, val_preds)
        val_metrics[fold] = {"MSE": float(val_mse), "MAE": float(val_mae)}
        print(f"[{fold}] Validation MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")
        
        # Test Eval
        test_mse = mean_squared_error(y_test_real, test_preds)
        test_mae = mean_absolute_error(y_test_real, test_preds)
        test_metrics[fold] = {"MSE": float(test_mse), "MAE": float(test_mae)}
        
        # Save test predictions
        np.save(os.path.join(results_dir, "last_test_preds.npy"), test_preds)
        
        torch.save(model.state_dict(), os.path.join(results_dir, f"{fold}_model.pth"))
        
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    print("LSTM training complete!")

if __name__ == "__main__":
    train_lstm()
