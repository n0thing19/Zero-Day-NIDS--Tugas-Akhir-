"""
Evaluation, metrics computation, and visualization utilities for NIDS models.
"""

from pathlib import Path
from typing import Dict, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    roc_auc = None
    pr_auc = None
    if y_scores is not None and len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_scores))
        pr_auc = float(average_precision_score(y_true, y_scores))

    return {
        "threshold": threshold,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "FPR": fpr,
        "FNR": fnr,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }


def evaluate_pipeline_models(
    y_true: np.ndarray,
    scores_dict: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    tau_sae = thresholds["tau_sae"]
    tau_if = thresholds["tau_iforest"]
    tau_max = thresholds.get("tau_max_normalized", 1.0)
    tau_w = thresholds.get("tau_weighted_average", 0.5)

    sae_norm = scores_dict["sae_norm"]
    if_norm = scores_dict["iforest_norm"]
    max_norm = scores_dict.get("hybrid_max_norm", np.maximum(sae_norm / tau_sae, if_norm / tau_if))
    weighted = scores_dict.get("hybrid_weighted", 0.5 * sae_norm + 0.5 * if_norm)

    pred_sae = (sae_norm > tau_sae).astype(int)
    pred_if = (if_norm > tau_if).astype(int)
    pred_or = (pred_sae | pred_if).astype(int)
    pred_max = (max_norm > tau_max).astype(int)
    pred_w = (weighted > tau_w).astype(int)

    results = {
        "SAE Alone": compute_classification_metrics(y_true, pred_sae, sae_norm, tau_sae),
        "IForest Alone": compute_classification_metrics(y_true, pred_if, if_norm, tau_if),
        "Hybrid (OR Fusion)": compute_classification_metrics(y_true, pred_or, max_norm, None),
        "Hybrid (Max Normalized)": compute_classification_metrics(y_true, pred_max, max_norm, tau_max),
        "Hybrid (Weighted Avg)": compute_classification_metrics(y_true, pred_w, weighted, tau_w),
    }

    df = pd.DataFrame(results).T
    cols = ["TP", "FP", "TN", "FN", "FNR", "FPR", "Recall", "Precision", "F1", "ROC_AUC", "PR_AUC"]
    return df[cols]


def plot_roc_pr_curves(
    y_true: np.ndarray,
    scores_dict: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_val = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_val:.4f})")

        p_vals, r_vals, _ = precision_recall_curve(y_true, scores)
        pr_val = average_precision_score(y_true, scores)
        axes[1].plot(r_vals, p_vals, label=f"{name} (AP={pr_val:.4f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random")
    axes[0].set_xlabel("False Positive Rate (FPR)")
    axes[0].set_ylabel("True Positive Rate (TPR / Recall)")
    axes[0].set_title("ROC Curve Comparison")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve Comparison")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrices(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions_dict.items()):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

        ax.set_title(f"{name}\nFNR={fnr:.2%} | FPR={fpr:.2%}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign (0)", "Attack (1)"])
        ax.set_yticklabels(["Benign (0)", "Attack (1)"])

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
