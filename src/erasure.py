from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src import metrics, sisa, splits

EXEC5 = splits.EXECUTION_RUNS[:5]
IMAG = splits.IMAGERY_RUNS
N_COHORT, N_ENROLLED, N_IMPOSTOR = 43, 40, 20
COHORT_SWEEP = [10, 20, 30, 43]
EMB_DIM = 128
ENC_EPOCHS = 60
ENC_BATCH = 256
ENC_LR = 1e-3
COSFACE_S, COSFACE_M = 30.0, 0.20


class Encoder(nn.Module):
    def __init__(self, in_dim=320, hidden=256, emb=EMB_DIM, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, emb))

    def forward(self, x):
        return self.net(x)


class CosFaceHead(nn.Module):
    def __init__(self, emb, n_cls, s=COSFACE_S, m=COSFACE_M):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_cls, emb))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m

    def forward(self, e, labels):
        e = F.normalize(e, dim=1)
        W = F.normalize(self.W, dim=1)
        cos = e @ W.t()
        oh = F.one_hot(labels, cos.shape[1]).float()
        return self.s * (cos - self.m * oh)


class CEHead(nn.Module):
    def __init__(self, emb, n_cls):
        super().__init__()
        self.fc = nn.Linear(emb, n_cls)

    def forward(self, e, labels):
        return self.fc(e)


def train_encoder(Xtr, ytr, Xval, yval, n_cls, *, mode, device, seed):
    torch.manual_seed(seed)
    enc = Encoder().to(device)
    head = (CosFaceHead(EMB_DIM, n_cls) if mode == "cosface"
            else CEHead(EMB_DIM, n_cls)).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=ENC_LR)
    crit = nn.CrossEntropyLoss()
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    Xval_t = torch.as_tensor(Xval, dtype=torch.float32, device=device)
    yval_t = torch.as_tensor(yval, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)
    n = len(ytr)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(ENC_EPOCHS):
        enc.train(); head.train()
        idx = rng.permutation(n)
        for i in range(0, n, ENC_BATCH):
            bi = idx[i:i + ENC_BATCH]
            opt.zero_grad()
            loss = crit(head(enc(Xtr_t[bi]), ytr_t[bi]), ytr_t[bi])
            loss.backward(); opt.step()
        enc.eval(); head.eval()
        with torch.no_grad():
            vloss = crit(head(enc(Xval_t), yval_t), yval_t).item()
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in enc.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    if best_state is not None:
        enc.load_state_dict(best_state)
    enc.eval()
    return enc


def embed(enc, X, device, batch=8192):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
            out.append(F.normalize(enc(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def build_templates(E_emb, E_sid, enrolled):
    T = {}
    for u in enrolled:
        t = E_emb[E_sid == u].mean(0)
        T[int(u)] = t / (np.linalg.norm(t) + 1e-9)
    return T


def openset_eer_templates(T, test_emb, test_sid, imp_emb, imp_sid, enrolled,
                          snorm_cohort=None):
    C = snorm_cohort
    use_snorm = snorm_cohort is not None and len(snorm_cohort)
    eers = []
    for u in enrolled:
        tu = T[int(u)]
        gen = test_emb[test_sid == u] @ tu
        imp = np.concatenate([test_emb[test_sid != u] @ tu, imp_emb @ tu])
        if use_snorm:
            tc = C @ tu
            mt, st = tc.mean(), tc.std() + 1e-9
            gen_probe = test_emb[test_sid == u]
            imp_probe = np.concatenate([test_emb[test_sid != u], imp_emb], axis=0)

            def snorm(scores, probes, tu_mean, tu_std):
                zc = probes @ C.T
                mz, sz = zc.mean(1), zc.std(1) + 1e-9
                return 0.5 * ((scores - tu_mean) / tu_std + (scores - mz) / sz)

            gen = snorm(gen, gen_probe, mt, st)
            imp = snorm(imp, imp_probe, mt, st)
        eers.append(metrics.compute_eer(gen, imp)[0])
    return float(np.mean(eers))


def openset_eer_sisa(consts, mode, Xte, sid_te, Xop, classes, device):
    Mt = sisa.score_matrix(consts, mode, Xte, classes, device)
    Mo = sisa.score_matrix(consts, mode, Xop, classes, device)
    eers = []
    for j, s in enumerate(classes):
        gen = Mt[sid_te == s, j]
        if len(gen) == 0:
            continue
        eers.append(metrics.compute_eer(gen, np.concatenate([Mt[sid_te != s, j], Mo[:, j]]))[0])
    return float(np.mean(eers))


def three_way_split(all_subjects, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(all_subjects)
    cohort = np.array(sorted(perm[:N_COHORT].tolist()))
    enrolled = np.array(sorted(perm[N_COHORT:N_COHORT + N_ENROLLED].tolist()))
    impostor = np.array(sorted(perm[N_COHORT + N_ENROLLED:
                                    N_COHORT + N_ENROLLED + N_IMPOSTOR].tolist()))
    return cohort, enrolled, impostor
