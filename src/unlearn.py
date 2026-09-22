from __future__ import annotations

import time

import numpy as np

from src import model

def find_shard_slice(shards, subject_to_shard, subject):
    k = subject_to_shard[int(subject)]
    for si, subs in enumerate(shards[k]["slice_subjects"]):
        if int(subject) in {int(s) for s in subs}:
            return k, si
    raise ValueError(f"subject {subject} not found in its shard's slices")

def train_shard_sliced(shard, Xtr, sid_tr, Xval, sid_val, *, exclude=None,
                       start_slice=0, init_state=None, device=None, seed=None,
                       epochs=None, patience=5):
    classes = shard["classes"]
    vmask = np.isin(sid_val, classes)
    Xv = Xval[vmask]
    yv = model.encode_labels(sid_val[vmask], classes)

    state, net = init_state, None
    cost_steps, epochs_list = 0, []
    slice_states: dict[int, dict] = {}
    cum: list[np.ndarray] = []
    t0 = time.time()
    for i, rows in enumerate(shard["slice_rows"]):
        if exclude is not None:
            rows = rows[sid_tr[rows] != int(exclude)]
        cum.append(rows)
        if i < start_slice:
            continue
        r = np.concatenate(cum)
        Xc = Xtr[r]
        yc = model.encode_labels(sid_tr[r], classes)
        net, info = model.train_mlp(Xc, yc, Xv, yv, len(classes),
                                    init_state=state, device=device, seed=seed,
                                    epochs=epochs, patience=patience,
                                    return_info=True)
        state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        slice_states[i] = state
        cost_steps += info["epochs_run"] * info["n_samples"]
        epochs_list.append(info["epochs_run"])
    return {"model": net, "classes": classes, "slice_states": slice_states,
            "cost_steps": int(cost_steps), "wall_s": time.time() - t0,
            "epochs": epochs_list}

def build_constituents(shards, Xtr, sid_tr, Xval, sid_val, *, device, seed,
                       epochs=None, patience=5):
    out = []
    total_cost, t0 = 0, time.time()
    for sh in shards:
        res = train_shard_sliced(sh, Xtr, sid_tr, Xval, sid_val, start_slice=0,
                                 init_state=None, device=device, seed=seed,
                                 epochs=epochs, patience=patience)
        out.append(res)
        total_cost += res["cost_steps"]
    return out, {"cost_steps": int(total_cost), "wall_s": time.time() - t0}

def unlearn_subject_aware(shards, subject_to_shard, before, X, Xtr, sid_tr,
                          Xval, sid_val, *, device, seed, epochs=None, patience=5):
    k, sx = find_shard_slice(shards, subject_to_shard, X)
    init = before[k]["slice_states"][sx - 1] if sx > 0 else None
    res = train_shard_sliced(shards[k], Xtr, sid_tr, Xval, sid_val, exclude=X,
                             start_slice=sx, init_state=init, device=device, seed=seed,
                             epochs=epochs, patience=patience)
    res["shard"], res["slice"] = k, sx
    return res

def retrain_all_minus(shards, X, Xtr, sid_tr, Xval, sid_val, *, device, seed,
                      epochs=None, patience=5):
    consts, total_cost, t0 = [], 0, time.time()
    for sh in shards:
        res = train_shard_sliced(sh, Xtr, sid_tr, Xval, sid_val, exclude=X,
                                 start_slice=0, init_state=None, device=device, seed=seed,
                                 epochs=epochs, patience=patience)
        consts.append(res)
        total_cost += res["cost_steps"]
    return consts, {"cost_steps": int(total_cost), "wall_s": time.time() - t0,
                    "n_shards_retrained": len(shards)}
