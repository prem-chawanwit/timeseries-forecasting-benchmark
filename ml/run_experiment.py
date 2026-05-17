import os
import subprocess
import datetime
import json
import argparse
import sys
# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_root)

def main():
    parser = argparse.ArgumentParser(description="Run the forecasting benchmark pipeline")
    parser.add_argument("--run_name", type=str, help="Optional specific run name")
    args = parser.parse_args()

    # 1. Create a unique run folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else f"run_{timestamp}"
    exp_dir = os.path.join("experiments", run_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 2. Setup environment variable so train/test scripts know where to save
    os.environ["EXPERIMENT_DIR"] = exp_dir
    
    import shutil
    if os.path.exists("ml/config.json"):
        shutil.copy("ml/config.json", os.path.join(exp_dir, "config_used.json"))
    
    print(f"==================================================")
    print(f"Starting Experiment: {run_name}")
    print(f"Results will be saved to: {exp_dir}")
    print(f"==================================================\n")
    
    # 3. Write initial metadata
    metadata = {
        "run_id": run_name,
        "timestamp": timestamp,
        "status": "running"
    }
    
    meta_path = os.path.join(exp_dir, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    # 4. Run scripts
    scripts = [
        "ml/src/etl/process_and_split.py",
        "ml/src/etl/feature_selection.py",
        "ml/src/train/baseline/train.py",
        "ml/src/train/xgboost/train.py",
        "ml/src/train/lstm/train.py",
        "ml/src/test/evaluate.py"
    ]
    
    for script in scripts:
        print(f"\n>>> Running {script}...")
        result = subprocess.run(["python", script])
        if result.returncode != 0:
            print(f"Error running {script}")
            metadata["status"] = "failed"
            break
    else:
        metadata["status"] = "completed"
        
    # Update metadata status
    metadata["end_time"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\n==================================================")
    print(f"Experiment '{run_name}' {metadata['status']}!")
    print(f"Check '{exp_dir}' for your benchmark results, models, and graphs.")
    print(f"==================================================")

if __name__ == "__main__":
    main()
