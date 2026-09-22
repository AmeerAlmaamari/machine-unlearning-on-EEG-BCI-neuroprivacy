from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src import config

class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int,
                 hidden=tuple(config.MLP_HIDDEN), dropout: float = 0.3):
        super().__init__()
        dims = [in_dim, *hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-1], n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def encode_labels(subject_ids: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.searchsorted(classes, subject_ids)

def train_mlp(Xtr, ytr, Xval, yval, n_classes, *, epochs=None, batch=None,
              lr=None, patience=5, device=None, seed=None, init_state=None,
              return_info=False):
    epochs = epochs or config.MLP_EPOCHS
    batch = batch or config.MLP_BATCH_SIZE
    lr = lr or config.MLP_LR
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)

    model = MLP(Xtr.shape[1], n_classes).to(device)
    if init_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in init_state.items()})
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    Xval_t = torch.as_tensor(Xval, dtype=torch.float32, device=device)
    yval_t = torch.as_tensor(yval, dtype=torch.long, device=device)

    rng = np.random.default_rng(seed)
    n = len(ytr)
    best, best_state, bad, epochs_run = float("inf"), None, 0, 0
    for _ in range(epochs):
        epochs_run += 1
        model.train()
        idx = rng.permutation(n)
        for i in range(0, n, batch):
            bi = idx[i:i + batch]
            opt.zero_grad()
            loss = crit(model(Xtr_t[bi]), ytr_t[bi])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xval_t), yval_t).item()
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if return_info:
        return model, {"epochs_run": epochs_run, "n_samples": n}
    return model

def predict_proba(model, X, device=None, batch=8192) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
            out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)

def embed(model, X, device=None, batch=8192) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
            out.append(model.net[:-1](xb).cpu().numpy())
    return np.concatenate(out, axis=0)
