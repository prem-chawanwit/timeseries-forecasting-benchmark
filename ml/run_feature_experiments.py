import os
import subprocess
import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Feature Ablation Experiments")
    parser.add_argument("--n_times", type=int, default=1, help="Number of times to run EACH feature set")
    args = parser.parse_args()

    # Reasonable assumptions for feature combinations
    feature_sets = {
        "all_features": [
            'TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 
            'Motor_current', 'COMP', 'DV_eletric', 'Towers', 'MPG', 'LPS', 
            'Pressure_switch', 'Oil_level', 'Caudal_impulses'
        ],
        "top_4_correlated": ['TP2', 'COMP', 'Motor_current', 'Oil_temperature'],
        "top_8_correlated": ['TP2', 'COMP', 'Motor_current', 'Oil_temperature', 'DV_eletric', 'DV_pressure', 'Towers', 'H1'],
        "only_target_history": ['TP2']
    }

    base_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting Feature Ablation Experiments")
    print(f"Will run each of the {len(feature_sets)} feature sets {args.n_times} times.")
    
    total_runs = len(feature_sets) * args.n_times
    current_run = 1

    for exp_name, features in feature_sets.items():
        for i in range(args.n_times):
            print(f"\n{'='*60}")
            print(f"Run {current_run}/{total_runs} | Experiment: {exp_name} (Iteration {i+1}/{args.n_times})")
            print(f"Features: {features}")
            print(f"{'='*60}")
            
            # Export features to env so train scripts can read it
            os.environ["FEATURE_COLS"] = ",".join(features)
            
            # Run the main experiment pipeline with a specific run name
            full_run_name = f"ablation_{exp_name}_iter{i+1}_{base_timestamp}"
            
            # Use subprocess to call run_experiment.py
            result = subprocess.run(["python", "run_experiment.py", "--run_name", full_run_name])
            
            if result.returncode != 0:
                print(f"Warning: Run {full_run_name} failed!")
            
            current_run += 1

    print("\n==================================================")
    print("All feature ablation experiments completed successfully!")
    print("Check the 'experiments/' directory for 'ablation_*' folders.")
    print("==================================================")

if __name__ == "__main__":
    main()
