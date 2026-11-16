"""API dự báo lưu lượng – phục vụ mô hình GRU/LSTM đã huấn luyện bằng Federated Learning.

Chạy thử trên máy:   uvicorn main:app --port 8080        (trong thư mục api/)
Deploy Cloud Run:    xem cloud/deploy_api.ps1 hoặc cloud/deploy_api.sh

Endpoint:
  GET /health                         – kiểm tra sống + tên region (dùng đo độ trễ)
  GET /cells                          – danh sách trạm (Edge node): mã ô, khu vực, toạ độ lưới
  GET /predict?cell_index=0&split=test – dự báo toàn bộ cửa sổ của 1 trạm, kèm giá trị thật, RMSE, MAE
  POST /predict                       – dự báo 1 cửa sổ do client gửi lên (24 giờ × 6 đặc trưng đã chuẩn hoá)
  GET /experiments                    – kết quả các thí nghiệm FL (RMSE theo vòng, dung lượng truyền)
"""
import glob, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE / "data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE / "model"))
REGION = os.environ.get("REGION", "local")
torch.set_num_threads(1)


class RNNForecaster(nn.Module):
    """Giống hệt fl5g/flcore.py để nạp được state_dict."""
    def __init__(self, n_feat=6, hidden=64, layers=2, T=1, cell="gru", dropout=0.1):
        super().__init__()
        R = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = R(n_feat, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, T)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


# ---------- Nạp dữ liệu + mô hình khi khởi động ----------
META = json.load(open(DATA_DIR / "meta.json", encoding="utf-8"))
NPZ = np.load(DATA_DIR / "clients.npz")
CFG = json.load(open(MODEL_DIR / "model_config.json", encoding="utf-8"))
MODEL = RNNForecaster(hidden=int(CFG["hidden"]), layers=int(CFG["layers"]), T=int(META["T"]), cell=CFG["cell"])
MODEL.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu"))
MODEL.eval()
EXPERIMENTS = []
for f in sorted(glob.glob(str(DATA_DIR / "results" / "*.json"))):
    r = json.load(open(f, encoding="utf-8"))
    ev = [e for e in r.get("eval", []) if e.get("round", 0) > 0]
    cum, ups = 0, []
    for e in ev:
        cum += next((x["uplink_bytes"] for x in r.get("rounds", []) if x["round"] == e["round"]), 0)
        ups.append(round(cum / 1e6, 4))
    EXPERIMENTS.append({"tag": r["tag"], "final_rmse": r.get("final_rmse"),
                        "rmse_dan_cu": ev[-1].get("rmse_dan_cu") if ev else None,
                        "rmse_van_phong": ev[-1].get("rmse_van_phong") if ev else None,
                        "uplink_MB": r.get("total_uplink_MB"), "downlink_MB": r.get("total_downlink_MB"),
                        "rounds": [e["round"] for e in ev], "rmse": [e["rmse"] for e in ev], "uplink_cum_MB": ups})

app = FastAPI(title="FL 5G Traffic Forecast API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _client(i):
    if not 0 <= i < len(META["clients"]):
        raise HTTPException(404, f"cell_index phải trong khoảng 0..{len(META['clients']) - 1}")
    return META["clients"][i]


def _inverse(z, c):
    """Giá trị chuẩn hoá -> đơn vị gốc của dữ liệu Milan."""
    return np.expm1(np.asarray(z) * c["sd"] + c["mu"]).clip(min=0)


@app.get("/health")
def health():
    return {"status": "ok", "region": REGION, "model": CFG.get("tag", "fedavg"), "n_cells": len(META["clients"])}


@app.get("/cells")
def cells():
    return [{"index": i, "cell": c["cell"], "group": c["group"], "row": (c["cell"] - 1) // 100,
             "col": (c["cell"] - 1) % 100, "n_train": c.get("n_train")} for i, c in enumerate(META["clients"])]


@app.get("/predict")
def predict_series(cell_index: int = 0, split: str = "test"):
    if split not in ("train", "val", "test"):
        raise HTTPException(400, "split phải là train | val | test")
    c = _client(cell_index)
    X = torch.tensor(NPZ[f"{split}_X_{cell_index}"]); Y = NPZ[f"{split}_Y_{cell_index}"]
    t0 = time.perf_counter()
    with torch.no_grad():
        P = MODEL(X).numpy()
    infer_ms = (time.perf_counter() - t0) * 1000
    rmse = float(math.sqrt(((P - Y) ** 2).mean()))
    pred, actual = _inverse(P[:, 0], c), _inverse(Y[:, 0], c)
    return {"region": REGION, "cell": c["cell"], "group": c["group"], "split": split, "n": int(len(Y)),
            "rmse_norm": rmse, "mae": float(np.abs(pred - actual).mean()),
            "inference_ms": round(infer_ms, 3), "predicted": pred.round(3).tolist(), "actual": actual.round(3).tolist()}


class Window(BaseModel):
    cell_index: int
    window: list[list[float]]     # 24 dòng × 6 cột đặc trưng đã chuẩn hoá (giống dữ liệu huấn luyện)


@app.post("/predict")
def predict_window(w: Window):
    c = _client(w.cell_index)
    x = torch.tensor([w.window], dtype=torch.float32)
    if x.shape[1:] != (META["L"], 6):
        raise HTTPException(400, f"window phải có kích thước {META['L']} x 6")
    t0 = time.perf_counter()
    with torch.no_grad():
        p = MODEL(x).numpy()[0]
    return {"region": REGION, "cell": c["cell"], "prediction_norm": p.tolist(),
            "prediction": _inverse(p, c).round(3).tolist(), "inference_ms": round((time.perf_counter() - t0) * 1000, 3)}


@app.get("/experiments")
def experiments():
    return EXPERIMENTS
