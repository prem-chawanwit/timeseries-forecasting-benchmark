import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_tabular_data(df, feature_cols, target_col):
    """ Prepare data for Baseline and XGBoost """
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y

def create_sequences(data, target, seq_length):
    """ Prepare sequences for LSTM """
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = target[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
