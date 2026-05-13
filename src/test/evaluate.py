import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import itertools

def evaluate_all():
    print("=== Benchmark Evaluation Report ===")
    
    results_base = os.environ.get("EXPERIMENT_DIR", "results")
    
    models = ["baseline", "xgboost", "lstm"]
    all_results = []
    
    for model in models:
        metrics_path = os.path.join(results_base, model, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                
            # Average across folds
            avg_mse = sum([m["MSE"] for m in metrics.values()]) / len(metrics)
            avg_mae = sum([m["MAE"] for m in metrics.values()]) / len(metrics)
            
            all_results.append({
                "Model": model.capitalize(),
                "Average MSE": avg_mse,
                "Average MAE": avg_mae
            })
            
    if all_results:
        df_results = pd.DataFrame(all_results)
        print("\n--- Summary ---")
        print(df_results.to_markdown(index=False))
        
        df_results.to_csv(os.path.join(results_base, "benchmark_summary.csv"), index=False)
        print("\nSaved benchmark summary to results/benchmark_summary.csv")
        
        # --- Plotting the Benchmark Comparison ---
        plot_benchmark(df_results, results_base)
        plot_time_series(results_base)
        
        # --- Statistical Testing ---
        perform_statistical_testing(results_base, models)
        
    else:
        print("No results found. Please run the training scripts first.")

def perform_statistical_testing(save_dir, models):
    print("\n=== Statistical Significance Test (K-Fold Paired t-test) ===")
    print("Testing if differences between models are statistically significant based on Fold results.\n")
    
    # Load fold metrics
    all_fold_mse = {}
    for model in models:
        metrics_path = os.path.join(save_dir, model, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                # Sort by fold to ensure alignment
                sorted_folds = sorted(metrics.keys())
                all_fold_mse[model] = [metrics[fold]["MSE"] for fold in sorted_folds]
                
    # Compare each pair of models
    pairs = list(itertools.combinations(models, 2))
    stat_results = []
    
    for m1, m2 in pairs:
        if m1 in all_fold_mse and m2 in all_fold_mse:
            mse_m1 = all_fold_mse[m1]
            mse_m2 = all_fold_mse[m2]
            
            # Two-tailed Paired t-test (Are they different?)
            t_stat_2, p_val_2 = stats.ttest_rel(mse_m1, mse_m2)
            
            # One-tailed Paired t-test (Is one strictly better than the other?)
            # Scipy stats.ttest_rel returns two-tailed. One tailed p-value is p_val_2 / 2
            # We want to know if m1 is less than m2, or m2 is less than m1
            t_stat_1_m1_better, p_val_1_m1_better = stats.ttest_rel(mse_m1, mse_m2, alternative='less')
            t_stat_1_m2_better, p_val_1_m2_better = stats.ttest_rel(mse_m2, mse_m1, alternative='less')
            
            winner = None
            p_val_winner = None
            if p_val_1_m1_better < 0.05:
                winner = m1
                p_val_winner = p_val_1_m1_better
            elif p_val_1_m2_better < 0.05:
                winner = m2
                p_val_winner = p_val_1_m2_better
                
            stat_results.append({
                "Model A": m1.capitalize(),
                "Model B": m2.capitalize(),
                "Two-Tail P-Value": p_val_2,
                "Significant Diff? (p<0.05)": "Yes" if p_val_2 < 0.05 else "No",
                "Winner (One-Tail)": winner.capitalize() if winner else "None (Tie)",
                "One-Tail P-Value": p_val_winner if winner else "-"
            })
            
    df_stats = pd.DataFrame(stat_results)
    stats_csv_path = os.path.join(save_dir, "statistical_tests.csv")
    print(df_stats.to_markdown(index=False))
    df_stats.to_csv(stats_csv_path, index=False)
    print(f"\nSaved statistical tests to {stats_csv_path}")
    print("Note: Since N=3 (3 Folds), the statistical power is low. P-values might be high unless differences are massive.")

def plot_benchmark(df, save_dir):
    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df["Model"]))
    width = 0.35
    
    # Create bar plots for MSE and MAE
    rects1 = ax1.bar(x - width/2, df["Average MSE"], width, label='MSE', color='skyblue')
    rects2 = ax1.bar(x + width/2, df["Average MAE"], width, label='MAE', color='salmon')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_ylabel('Error Value')
    ax1.set_title('Forecasting Benchmark: Model Comparison on Global Test Set')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model"])
    ax1.legend()
    
    # Add value labels on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    
    plot_path = os.path.join(save_dir, "benchmark_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved benchmark plot to {plot_path}")

def plot_time_series(save_dir, num_points=200):
    exp_dir = os.environ.get("EXPERIMENT_DIR", "")
    base_data_dir = os.path.join(exp_dir, "data") if exp_dir else "data"
    splits_dir = os.path.join(base_data_dir, "3_kfold_splits")
    
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    
    true_values = test_df['target'].values
    
    plt.figure(figsize=(14, 6))
    
    # Plot True Values (first num_points)
    plt.plot(true_values[:num_points], label='True Target (TP2)', color='black', linewidth=2, linestyle='--')
    
    colors = {'baseline': 'blue', 'xgboost': 'green', 'lstm': 'red'}
    
    models = ["baseline", "xgboost", "lstm"]
    for model in models:
        pred_path = os.path.join(save_dir, model, "last_test_preds.npy")
        if os.path.exists(pred_path):
            preds = np.load(pred_path)
            
            # Note: LSTM might have fewer predictions because it drops the first `seq_length` rows.
            # To align them, we align from the end: True[-len(preds):] matches Preds
            # For plotting first 200 points, we need to map correctly.
            offset = len(true_values) - len(preds)
            
            # Plot from index `offset` to `offset + num_points` if we wanted exact alignment,
            # but to make it simple, we just plot the first num_points of preds with an offset on x-axis
            x_axis = np.arange(offset, offset + min(num_points, len(preds)))
            
            plt.plot(x_axis, preds[:len(x_axis)], label=f'{model.capitalize()} Pred', color=colors[model], alpha=0.7)

    plt.title(f'Time Series Forecast Comparison (First {num_points} points of Test Set)')
    plt.xlabel('Time Step')
    plt.ylabel('TP2 Target Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ts_path = os.path.join(save_dir, "timeseries_comparison.png")
    plt.savefig(ts_path, dpi=300)
    print(f"Saved Time Series plot to {ts_path}")

if __name__ == "__main__":
    evaluate_all()
