"""Dữ liệu + mô hình dùng chung cho ClientApp và ServerApp.

Dữ liệu client được chuẩn bị trước bằng prepare_data.py và lưu ở thư mục FL5G_DATA
(mặc định: ~/fl5g_data). Mỗi SuperNode (partition-id = i) là 1 trạm phát (1 ô lưới Milan).
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

from fl5g import flcore as fc

DATA_DIR = Path(os.environ.get("FL5G_DATA", Path.home() / "fl5g_data"))
OUT_DIR = Path(os.environ.get("FL5G_OUT", Path.home() / "fl5g_results"))
_cache = {}


def load_clients():
    """Trả về (clients, meta). clients[i] = dict(train/val/test=(X, Y), group, cell, mu, sd) – tensor CPU."""
    if "clients" not in _cache:
        meta = json.load(open(DATA_DIR / "meta.json", encoding="utf-8"))
        z = np.load(DATA_DIR / "clients.npz")
        clients = []
        for i, info in enumerate(meta["clients"]):
            c = {k: (torch.tensor(z[f"{k}_X_{i}"]), torch.tensor(z[f"{k}_Y_{i}"])) for k in ("train", "val", "test")}
            c.update(info)
            clients.append(c)
        _cache["clients"], _cache["meta"] = clients, meta
    return _cache["clients"], _cache["meta"]


def make_model(cfg):
    _, meta = load_clients()
    return fc.RNNForecaster(n_feat=6, hidden=int(cfg["hidden"]), layers=int(cfg["layers"]),
                            T=int(meta["T"]), cell=str(cfg["cell"]))


def train_local(model, data, cfg, mu=0.0):
    """Huấn luyện cục bộ (FedAvg nếu mu = 0, FedProx nếu mu > 0)."""
    task = fc.RegTask("test")
    x, y = data
    model.train()
    w0 = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning-rate"]))
    n, bs, total, steps = len(y), int(cfg["batch-size"]), 0.0, 0
    for _ in range(int(cfg["local-epochs"])):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = task.loss(model(x[b]), y[b])
            if mu > 0:
                loss = loss + mu / 2 * sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), w0))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item(); steps += 1
    return total / max(steps, 1)


def evaluate_all(model, split="test"):
    """RMSE (chuẩn hoá) trên toàn bộ trạm + theo từng khu vực."""
    clients, _ = load_clients()
    res = fc.RegTask(split).evaluate(model, clients, per_client=True)
    out = {"rmse": res["rmse"], "mae_orig": res["mae_orig"]}
    for g in sorted({c["group"] for c in clients}):
        v = [p["rmse"] for p in res["per_client"] if p["group"] == g]
        key = "rmse_" + ("dan_cu" if g.startswith("Dân") else "van_phong" if g.startswith("Văn") else "khac")
        out[key] = float(np.mean(v))
    return out
