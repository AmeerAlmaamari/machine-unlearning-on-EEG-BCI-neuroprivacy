from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src import config, model

def shard_tag(mode: str, S: int, R: int, seed: int) -> str:
    return f"{mode}_S{S}_R{R}_seed{seed}"

def make_shards(sid_tr: np.ndarray, S: int, R: int, mode: str, seed: int):
    rng = np.random.default_rng(seed)
    all_subjects = np.array(sorted(np.unique(sid_tr)))
    shards: list[dict] = []
    subject_to_shard: dict[int, int] = {}

    if mode == "subject_aware":
        groups = np.array_split(rng.permutation(all_subjects), S)
        for k, grp in enumerate(groups):
            grp = np.array(sorted(grp.tolist()))
            n_slices = min(R, len(grp))
            subj_slices = [np.array(sorted(s.tolist()))
                           for s in np.array_split(rng.permutation(grp), n_slices)]
            slice_rows = [np.where(np.isin(sid_tr, ss))[0] for ss in subj_slices]
            shards.append({"classes": grp, "slice_rows": slice_rows,
                           "slice_subjects": subj_slices})
            for s in grp:
                subject_to_shard[int(s)] = k

    elif mode == "uniform":
        assign = rng.integers(0, S, size=len(sid_tr))
        for k in range(S):
            rows = np.where(assign == k)[0]
            n_slices = min(R, len(rows))
            slice_rows = [np.array(sorted(s.tolist()))
                          for s in np.array_split(rng.permutation(rows), n_slices)]
            shards.append({"classes": all_subjects, "slice_rows": slice_rows,
                           "slice_subjects": None})
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return shards, subject_to_shard

def assert_subject_aware_partition(shards, all_subjects) -> None:
    seen = np.concatenate([sh["classes"] for sh in shards])
    assert len(seen) == len(np.unique(seen)), "a subject appears in >1 shard"
    assert set(seen.tolist()) == set(int(s) for s in all_subjects), "subjects missing"

def train_constituents(shards, Xtr, sid_tr, Xval, sid_val, *, device, seed, ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    constituents = []
    for k, sh in enumerate(shards):
        classes = sh["classes"]
        vmask = np.isin(sid_val, classes)
        Xv = Xval[vmask]
        yv = model.encode_labels(sid_val[vmask], classes)

        state, net, ckpts, cum = None, None, [], []
        for i, rows in enumerate(sh["slice_rows"]):
            cum.append(rows)
            r = np.concatenate(cum)
            Xc = Xtr[r]
            yc = model.encode_labels(sid_tr[r], classes)
            net = model.train_mlp(Xc, yc, Xv, yv, len(classes),
                                  init_state=state, device=device, seed=seed)
            state = {kk: vv.detach().clone() for kk, vv in net.state_dict().items()}
            p = ckpt_dir / f"shard{k:02d}_slice{i + 1}.pt"
            torch.save({"state": {kk: vv.cpu() for kk, vv in state.items()},
                        "classes": classes}, p)
            ckpts.append(str(p))
        constituents.append({"model": net, "classes": classes, "ckpts": ckpts})
    return constituents

def load_checkpoint(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    classes = blob["classes"]
    net = model.MLP(config.N_FEATURES, len(classes)).to(device)
    net.load_state_dict({k: v.to(device) for k, v in blob["state"].items()})
    return net, classes

def score_matrix(constituents, mode, X, all_classes, device) -> np.ndarray:
    n = len(X)
    col_of = {int(s): i for i, s in enumerate(all_classes)}
    if mode == "subject_aware":
        M = np.zeros((n, len(all_classes)), dtype=np.float32)
        for c in constituents:
            probs = model.predict_proba(c["model"], X, device=device)
            for j, s in enumerate(c["classes"]):
                M[:, col_of[int(s)]] = probs[:, j]
        return M

    acc = np.zeros((n, len(all_classes)), dtype=np.float64)
    for c in constituents:
        probs = model.predict_proba(c["model"], X, device=device)
        for j, s in enumerate(c["classes"]):
            acc[:, col_of[int(s)]] += probs[:, j]
    return (acc / len(constituents)).astype(np.float32)
