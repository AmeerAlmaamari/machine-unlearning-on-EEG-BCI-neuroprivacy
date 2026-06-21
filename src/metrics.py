"""Biometric verification metrics: EER, TAR@FAR, ROC-AUC.

Verification convention: for a claimed subject X we score every test window by
the model's softmax probability of class X. Genuine = windows truly from X;
impostor = windows from any other subject (zero-effort impostors).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate and its threshold from genuine/impostor score arrays."""
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))   # FAR == FRR crossing
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thr[idx])


def tar_at_far(genuine: np.ndarray, impostor: np.ndarray, far: float = 0.01) -> float:
    """True Accept Rate at a target False Accept Rate."""
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(far, fpr, tpr))


def verification_report(probs: np.ndarray, subj_test: np.ndarray,
                        classes: np.ndarray, far: float = 0.01) -> dict:
    """Per-subject EER/AUC/TAR averaged over all enrolled subjects.

    probs   : [n_test, n_classes] softmax probabilities
    subj_test : [n_test] true subject id per row
    classes : [n_classes] subject id for each probability column
    """
    eers, aucs, tars = [], [], []
    per_subject = {}
    for j, sid in enumerate(classes):
        col = probs[:, j]
        gen = col[subj_test == sid]
        imp = col[subj_test != sid]
        if len(gen) == 0 or len(imp) == 0:
            continue
        eer, _ = compute_eer(gen, imp)
        labels = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
        auc = roc_auc_score(labels, np.concatenate([gen, imp]))
        tar = tar_at_far(gen, imp, far)
        eers.append(eer); aucs.append(auc); tars.append(tar)
        per_subject[int(sid)] = {"eer": float(eer), "auc": float(auc), "tar": float(tar)}

    # closed-set identification accuracy (argmax over columns)
    pred = classes[np.argmax(probs, axis=1)]
    acc = float(np.mean(pred == subj_test))

    eers = np.array(eers)
    return {
        "mean_eer": float(eers.mean()), "std_eer": float(eers.std()),
        "median_eer": float(np.median(eers)), "max_eer": float(eers.max()),
        "mean_auc": float(np.mean(aucs)),
        "mean_tar_at_far": float(np.mean(tars)), "far_target": far,
        "identification_accuracy": acc,
        "n_subjects_scored": int(len(eers)),
        "per_subject": per_subject,
    }


# --- self-test (known cases) -------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)
    # 1) perfectly separated -> EER = 0
    eer, _ = compute_eer(np.full(100, 0.9), np.full(100, 0.1))
    assert eer < 1e-9, eer
    # 2) identical large samples -> EER ~ 0.5
    g = rng.random(20000); i = rng.random(20000)
    eer2, _ = compute_eer(g, i)
    assert abs(eer2 - 0.5) < 0.02, eer2
    # 3) deterministic overlap -> EER = 0.5
    eer3, _ = compute_eer(np.array([0.6, 0.4]), np.array([0.6, 0.4]))
    assert abs(eer3 - 0.5) < 1e-9, eer3
    return eer, eer2, eer3


if __name__ == "__main__":
    print("EER self-test:", _selftest(), "OK")
