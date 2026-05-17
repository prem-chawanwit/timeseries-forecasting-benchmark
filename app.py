"""
ML Experiment Dashboard — Flask Backend
Serves API endpoints for config editing, experiment viewing, and experiment execution.
"""

import os
import json
import glob
import subprocess
import threading
from flask import Flask, render_template, jsonify, request, send_file, abort

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "ml", "config.json")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
RUN_SCRIPT = os.path.join(BASE_DIR, "ml", "run_experiment.py")

# Track running experiments
running_experiments = {}


# ─── Pages ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─── Config API ───────────────────────────────────────────────────────────────

@app.route("/api/config", methods=["GET"])
def get_config():
    """Read current ml/config.json"""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        return jsonify(config)
    except FileNotFoundError:
        return jsonify({"error": "config.json not found"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in config.json"}), 500


@app.route("/api/config", methods=["POST"])
def save_config():
    """Update ml/config.json with new values"""
    try:
        new_config = request.get_json()
        if not new_config:
            return jsonify({"error": "No JSON body provided"}), 400

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(new_config, f, indent=4, ensure_ascii=False)

        return jsonify({"success": True, "message": "Config saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Experiments API ──────────────────────────────────────────────────────────

@app.route("/api/experiments", methods=["GET"])
def list_experiments():
    """List all experiment runs with metadata"""
    experiments = []

    if not os.path.isdir(EXPERIMENTS_DIR):
        return jsonify([])

    for run_name in sorted(os.listdir(EXPERIMENTS_DIR), reverse=True):
        run_dir = os.path.join(EXPERIMENTS_DIR, run_name)
        if not os.path.isdir(run_dir):
            continue

        exp_info = {"run_id": run_name, "status": "unknown", "timestamp": ""}

        # Read run_meta.json if exists
        meta_path = os.path.join(run_dir, "run_meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                exp_info["status"] = meta.get("status", "unknown")
                exp_info["timestamp"] = meta.get("timestamp", "")
                exp_info["end_time"] = meta.get("end_time", "")
            except (json.JSONDecodeError, IOError):
                pass

        experiments.append(exp_info)

    return jsonify(experiments)


@app.route("/api/experiments/<run_id>", methods=["GET"])
def get_experiment(run_id):
    """Get detailed info for a specific experiment run"""
    run_dir = os.path.join(EXPERIMENTS_DIR, run_id)

    if not os.path.isdir(run_dir):
        return jsonify({"error": "Experiment not found"}), 404

    result = {
        "run_id": run_id,
        "config_used": None,
        "run_meta": None,
        "benchmark_summary": None,
        "statistical_tests": None,
        "images": [],
        "models": {},
        "etl_images": [],
    }

    # Config used
    config_path = os.path.join(run_dir, "config_used.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            result["config_used"] = json.load(f)

    # Run meta
    meta_path = os.path.join(run_dir, "run_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            result["run_meta"] = json.load(f)

    # Benchmark summary CSV → parse to list of dicts
    csv_path = os.path.join(run_dir, "benchmark_summary.csv")
    if os.path.isfile(csv_path):
        result["benchmark_summary"] = _parse_csv(csv_path)

    # Statistical tests CSV
    stats_path = os.path.join(run_dir, "statistical_tests.csv")
    if os.path.isfile(stats_path):
        result["statistical_tests"] = _parse_csv(stats_path)

    # Top-level PNG images
    for img_file in glob.glob(os.path.join(run_dir, "*.png")):
        result["images"].append(os.path.basename(img_file))

    # ETL images
    etl_dir = os.path.join(run_dir, "data", "2_etl")
    if os.path.isdir(etl_dir):
        for img_file in glob.glob(os.path.join(etl_dir, "*.png")):
            result["etl_images"].append(os.path.basename(img_file))

    # Selected features
    sel_feat_path = os.path.join(etl_dir, "selected_features.json")
    if os.path.isfile(sel_feat_path):
        with open(sel_feat_path, "r", encoding="utf-8") as f:
            result["selected_features"] = json.load(f)

    # Per-model data (baseline, xgboost, lstm)
    for model_name in ["baseline", "xgboost", "lstm"]:
        model_dir = os.path.join(run_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        model_info = {"metrics": None, "images": []}

        # Metrics
        metrics_path = os.path.join(model_dir, "metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                model_info["metrics"] = json.load(f)

        # Model-specific images (e.g., learning curves)
        for img_file in glob.glob(os.path.join(model_dir, "*.png")):
            model_info["images"].append(os.path.basename(img_file))

        result["models"][model_name] = model_info

    return jsonify(result)


@app.route("/api/experiments/<run_id>/images/<path:img_path>", methods=["GET"])
def serve_experiment_image(run_id, img_path):
    """Serve PNG images from experiment directories"""
    # img_path can be: "benchmark_comparison.png" or "lstm/learning_curve_fold_1.png"
    # or "data/2_etl/feature_ranking.png"
    file_path = os.path.join(EXPERIMENTS_DIR, run_id, img_path)

    if not os.path.isfile(file_path):
        abort(404)

    # Security: ensure file is within experiments directory
    real_path = os.path.realpath(file_path)
    real_exp = os.path.realpath(EXPERIMENTS_DIR)
    if not real_path.startswith(real_exp):
        abort(403)

    return send_file(real_path, mimetype="image/png")


# ─── Run Experiment API ──────────────────────────────────────────────────────

@app.route("/api/run", methods=["POST"])
def run_experiment():
    """Trigger run_experiment.py in a background thread"""
    # Check if an experiment is already running
    for exp_id, info in list(running_experiments.items()):
        if info.get("status") == "running":
            return jsonify({
                "error": "An experiment is already running",
                "running_id": exp_id
            }), 409

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"

    running_experiments[run_name] = {
        "status": "running",
        "output": "",
        "run_name": run_name,
    }

    def _run_in_background(name):
        try:
            proc = subprocess.Popen(
                ["python", RUN_SCRIPT, "--run_name", name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=BASE_DIR,
                bufsize=1,
            )
            output_lines = []
            for line in proc.stdout:
                output_lines.append(line)
                running_experiments[name]["output"] = "".join(output_lines[-200:])

            proc.wait()
            running_experiments[name]["status"] = (
                "completed" if proc.returncode == 0 else "failed"
            )
        except Exception as e:
            running_experiments[name]["status"] = "failed"
            running_experiments[name]["output"] += f"\nError: {str(e)}"

    thread = threading.Thread(target=_run_in_background, args=(run_name,), daemon=True)
    thread.start()

    return jsonify({
        "success": True,
        "run_name": run_name,
        "message": f"Experiment '{run_name}' started"
    })


@app.route("/api/run/status", methods=["GET"])
def run_status():
    """Check the status of running/recent experiments"""
    return jsonify(running_experiments)


@app.route("/api/run/status/<run_name>", methods=["GET"])
def run_status_detail(run_name):
    """Check status of a specific experiment run"""
    if run_name in running_experiments:
        return jsonify(running_experiments[run_name])
    return jsonify({"error": "Run not found"}), 404


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_csv(filepath):
    """Parse a simple CSV file into a list of dicts"""
    rows = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if not lines:
            return rows
        headers = lines[0].split(",")
        for line in lines[1:]:
            values = line.split(",")
            row = {}
            for i, h in enumerate(headers):
                row[h] = values[i] if i < len(values) else ""
            rows.append(row)
    except Exception:
        pass
    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  ML Experiment Dashboard")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
