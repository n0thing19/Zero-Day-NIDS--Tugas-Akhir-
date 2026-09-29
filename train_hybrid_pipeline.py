"""
End-to-end training and evaluation script for the Hybrid SAE + Cascaded IForest pipeline.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.config import PipelineConfig
from src.data_loader import (
    load_partition_metadata,
    load_scaler_params,
    load_evaluation_dataset,
)
from src.sae_model import load_sae, train_sae, save_sae, StackedAutoencoder
from src.iforest_model import CascadedIForest
from src.hybrid_fusion import HybridDetector
from src.evaluation import (
    evaluate_pipeline_models,
    plot_roc_pr_curves,
    plot_confusion_matrices,
)


def run_pipeline(
    train_sae_flag: bool = False,
    feature_mode: str = "latent_mse",
    fusion_strategy: str = "or_rule",
    threshold_percentile: float = 95.0,
):
    print("=" * 60, flush=True)
    print("HYBRID SAE + ISOLATION FOREST NIDS PIPELINE", flush=True)
    print("=" * 60, flush=True)

    config = PipelineConfig()
    config.resolve_paths()
    config.iforest_feature_mode = feature_mode
    config.fusion_strategy = fusion_strategy
    config.threshold_percentile = threshold_percentile

    # 1. Load Metadata & Scaler
    meta = load_partition_metadata(config.metadata_path)
    feature_columns = meta["feature_columns"]
    print(f"[1/6] Metadata loaded: {len(feature_columns)} features.", flush=True)

    mins, maxs = load_scaler_params(config.scaler_path)
    print(f"      Scaler loaded from: {config.scaler_path}", flush=True)

    # 2. SAE Model (Load or Train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2/6] Device: {device}", flush=True)

    train_files = sorted(config.train_dir.glob("*.parquet"))

    if train_sae_flag or not config.sae_weights_path.exists():
        print("      Training Stacked Autoencoder (SAE)...", flush=True)
        sae_model = StackedAutoencoder(
            input_dim=len(feature_columns),
            encoder_dims=config.encoder_dims,
            latent_dim=config.latent_dim,
            decoder_dims=config.decoder_dims,
            output_activation=config.output_activation,
        )
        history = train_sae(
            model=sae_model,
            train_files=train_files,
            feature_columns=feature_columns,
            mins=mins,
            maxs=maxs,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            device=device,
        )
        save_sae(
            model=sae_model,
            weights_path=config.sae_weights_path,
            config_path=config.sae_config_path,
            history=history,
            history_path=config.artifacts_dir / "sae_training_history.json",
        )
        print("      SAE training complete and saved.", flush=True)
    else:
        print(f"      Loading existing SAE weights from: {config.sae_weights_path}", flush=True)
        sae_model = load_sae(
            config.sae_weights_path,
            config.sae_config_path,
            device=device,
        )

    # 3. Cascaded Feature Extraction & IForest Training
    print(f"[3/6] Sampling train benign data for Cascaded IForest (mode='{feature_mode}')...", flush=True)
    X_train_benign, _, _ = load_evaluation_dataset(
        files=train_files,
        feature_columns=feature_columns,
        mins=mins,
        maxs=maxs,
        max_samples=config.max_train_samples_for_iforest,
        seed=config.seed,
    )
    print(f"      Collected {len(X_train_benign)} train samples.", flush=True)

    calib_size = int(len(X_train_benign) * config.iforest_calibration_ratio)
    X_fit = X_train_benign[calib_size:]
    X_calib = X_train_benign[:calib_size]

    print("      Training Cascaded Isolation Forest...", flush=True)
    cascaded_iforest = CascadedIForest(
        n_estimators=config.iforest_n_estimators,
        n_jobs=config.iforest_n_jobs,
        random_state=config.seed,
        feature_mode=feature_mode,
    )
    cascaded_iforest.fit(X_fit, sae_model)
    cascaded_iforest.save(config.iforest_model_path)
    print(f"      Cascaded IForest saved to: {config.iforest_model_path}", flush=True)

    # 4. Hybrid Normalization & Threshold Calibration
    print("[4/6] Calibrating thresholds on benign calibration data...", flush=True)
    hybrid_detector = HybridDetector(
        sae_model=sae_model,
        iforest_model=cascaded_iforest,
        fusion_strategy=fusion_strategy,
        alpha=config.alpha,
    )

    sae_train_raw = sae_model.get_reconstruction_error(X_train_benign)
    if_train_raw = cascaded_iforest.score_samples(X_train_benign, sae_model)
    hybrid_detector.fit_bounds(sae_train_raw, if_train_raw)

    sae_calib_raw = sae_model.get_reconstruction_error(X_calib)
    if_calib_raw = cascaded_iforest.score_samples(X_calib, sae_model)
    thresholds = hybrid_detector.calibrate_thresholds(
        sae_calib_raw,
        if_calib_raw,
        percentile=threshold_percentile,
    )
    hybrid_detector.save_state(config.artifacts_dir / "hybrid_detector_state.json")

    print(f"      Threshold Percentile : {threshold_percentile}%", flush=True)
    print(f"      Tau SAE              : {thresholds['tau_sae']:.6f}", flush=True)
    print(f"      Tau IForest          : {thresholds['tau_iforest']:.6f}", flush=True)
    print(f"      Tau Max Normalized   : {thresholds['tau_max_normalized']:.6f}", flush=True)
    print(f"      Tau Weighted Average : {thresholds['tau_weighted_average']:.6f}", flush=True)

    # 5. Evaluate on Test Partitions
    print("[5/6] Loading test partitions (Benign & Unseen Attack)...", flush=True)
    test_benign_files = sorted(config.test_benign_dir.glob("*.parquet"))
    unseen_attack_files = sorted(config.unseen_attack_dir.glob("*.parquet"))

    X_test_b, y_test_b, _ = load_evaluation_dataset(
        test_benign_files,
        feature_columns,
        mins,
        maxs,
        max_samples=config.eval_max_benign_samples,
        seed=config.seed,
    )
    X_test_a, y_test_a, _ = load_evaluation_dataset(
        unseen_attack_files,
        feature_columns,
        mins,
        maxs,
        max_samples=config.eval_max_attack_samples,
        seed=config.seed + 1,
    )

    X_test = np.vstack([X_test_b, X_test_a])
    y_test = np.concatenate([y_test_b, y_test_a])
    print(f"      Test Set: {len(X_test_b)} Benign, {len(X_test_a)} Attacks (Total: {len(X_test)})", flush=True)

    scores_dict = hybrid_detector.compute_scores(X_test)

    # 6. Comparative Evaluation
    print("\n[6/6] Computing Performance Metrics...", flush=True)
    results_df = evaluate_pipeline_models(y_test, scores_dict, thresholds)

    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)
    print("\n" + "=" * 80, flush=True)
    print("COMPARATIVE EVALUATION SUMMARY (AIM: LOWER FNR)", flush=True)
    print("=" * 80, flush=True)
    print(results_df.to_string(), flush=True)
    print("=" * 80, flush=True)

    # Save metrics
    metrics_path = config.results_dir / "hybrid_evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "thresholds": thresholds,
                "metrics": results_df.to_dict(orient="index"),
            },
            f,
            indent=4,
            default=str,
        )
    print(f"Metrics saved to: {metrics_path}", flush=True)

    # Plot Curves & Confusion Matrix
    curves_path = config.results_dir / "hybrid_roc_pr_curves.png"
    plot_roc_pr_curves(
        y_test,
        {
            "SAE Alone": scores_dict["sae_norm"],
            "Cascaded IForest": scores_dict["iforest_norm"],
            "Hybrid (Max-Norm)": scores_dict["hybrid_max_norm"],
        },
        save_path=curves_path,
        show=False,
    )
    print(f"ROC/PR curves saved to: {curves_path}", flush=True)

    cm_path = config.results_dir / "hybrid_confusion_matrices.png"
    tau_sae = thresholds["tau_sae"]
    tau_if = thresholds["tau_iforest"]
    pred_sae = (scores_dict["sae_norm"] > tau_sae).astype(int)
    pred_if = (scores_dict["iforest_norm"] > tau_if).astype(int)
    pred_or = (pred_sae | pred_if).astype(int)

    plot_confusion_matrices(
        y_test,
        {
            "SAE Alone": pred_sae,
            "Cascaded IForest": pred_if,
            "Hybrid (OR Fusion)": pred_or,
        },
        save_path=cm_path,
        show=False,
    )
    print(f"Confusion matrices saved to: {cm_path}", flush=True)
    print("\nPipeline execution finished successfully!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain-sae", action="store_true", help="Retrain SAE model")
    parser.add_argument("--feature-mode", type=str, default="latent_mse", choices=["latent_mse", "latent_residual", "latent_only"])
    parser.add_argument("--fusion-strategy", type=str, default="or_rule", choices=["or_rule", "max_normalized", "weighted_average"])
    parser.add_argument("--percentile", type=float, default=95.0)
    args = parser.parse_args()

    run_pipeline(
        train_sae_flag=args.retrain_sae,
        feature_mode=args.feature_mode,
        fusion_strategy=args.fusion_strategy,
        threshold_percentile=args.percentile,
    )
