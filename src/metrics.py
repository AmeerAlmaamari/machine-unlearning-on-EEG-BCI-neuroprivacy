from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def compute_eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thr[idx])

def tar_at_far(genuine: np.ndarray, impostor: np.ndarray, far: float = 0.01) -> float:
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(far, fpr, tpr))

def verification_report(probs: np.ndarray, subj_test: np.ndarray,
                        classes: np.ndarray, far: float = 0.01) -> dict:
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
