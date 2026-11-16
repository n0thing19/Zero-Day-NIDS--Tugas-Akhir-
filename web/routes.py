import random
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, render_template, request

from config import BENIGN_TEST_DIR, DEFAULT_TAU, UNSEEN_ATTACK_DIR
from inference_engine import ENGINE

api_bp = Blueprint("api", __name__)


def _get_default_port(category: str, is_attack: bool) -> int:
    if not is_attack:
        return 443
    port_map = {
        "Bot": 6667,
        "Brute Force": 22,
    }
    return port_map.get(category, 80)


def _compute_eval_status(is_attack: bool, is_anomaly: bool) -> str:
    if is_attack and is_anomaly:
        return "TP"
    if not is_attack and is_anomaly:
        return "FP"
    if is_attack and not is_anomaly:
        return "FN"
    return "TN"


@api_bp.route("/")
def index():
    return render_template("index.html")


@api_bp.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status": "connected",
        "device": str(ENGINE.device),
        "features": len(ENGINE.feature_columns),
        "model": ENGINE.config["model_name"],
        "tau_default": DEFAULT_TAU,
    })


@api_bp.route("/api/sample-flow", methods=["GET"])
def api_sample_flow():
    category = request.args.get("category", "Benign")
    tau = float(request.args.get("tau", DEFAULT_TAU))
    is_attack = category != "Benign"

    if not is_attack:
        files = list(BENIGN_TEST_DIR.glob("*.parquet"))
    else:
        folder = UNSEEN_ATTACK_DIR / category
        files = list(folder.glob("*.parquet")) if folder.exists() else []

    if not files:
        return jsonify({"error": "No files found for category"}), 404

    chosen_file = random.choice(files)
    df = pd.read_parquet(chosen_file, columns=ENGINE.feature_columns)
    row = df.sample(1).to_numpy(dtype=np.float32)[0]

    pred = ENGINE.predict_single_flow(row, tau=tau)
    is_anom = pred["verdict"] == "ANOMALY"

    return jsonify({
        "id": f"FLW-{random.randint(10000, 99999)}",
        "time": pd.Timestamp.now().strftime("%H:%M:%S"),
        "port": _get_default_port(category, is_attack),
        "groundTruth": category,
        "isAttack": is_attack,
        "evalStatus": _compute_eval_status(is_attack, is_anom),
        **pred,
    })


@api_bp.route("/api/explain-flow", methods=["POST"])
def api_explain_flow():
    data = request.json or {}
    features = np.array(data.get("features", []), dtype=np.float32)
    tau = float(data.get("tau", DEFAULT_TAU))

    if len(features) != len(ENGINE.feature_columns):
        return jsonify({"error": f"Invalid feature count. Expected {len(ENGINE.feature_columns)}"}), 400

    pred = ENGINE.predict_single_flow(features, tau=tau)
    return jsonify(pred)