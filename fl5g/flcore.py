"""
flcore.py – Thư viện Federated Learning dùng chung cho 2 bài:
  (1) Seminar: Non-IID trên MNIST / CIFAR-100, so sánh FedSGD / FedAvg / FedProx
  (2) Bài tập 19: Dự báo lưu lượng mạng 5G (Telecom Italia – Milan) với FL trên Edge + nén mô hình

Nội dung:
  1. Tiện ích chung (seed, thiết bị, Google Cloud Storage)
  2. Chia dữ liệu cho client (IID, Dirichlet, shards)
  3. Dữ liệu ảnh (MNIST, CIFAR-100)
  4. Dữ liệu chuỗi thời gian Milan (ETL, phân cụm khu vực, cửa sổ trượt)
  5. Mô hình (CNN cho ảnh, GRU/LSTM cho chuỗi thời gian)
  6. Task (phân loại / hồi quy): loss + đánh giá
  7. Nén cập nhật mô hình (fp16, int8, int4, top-k, top-k + int8) + error feedback
  8. Vòng lặp FL tổng quát: FedSGD, FedAvg, FedProx (+ straggler, + nén)
  9. Baseline: huấn luyện tập trung, huấn luyện cục bộ riêng lẻ
 10. Đo độ trễ Edge vs Cloud
 11. Vẽ biểu đồ, lập bảng, lưu kết quả
"""
import os, io, json, time, math, copy, random, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 1. Tiện ích chung
# =============================================================================
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def gcs_cp(src, dst, recursive=True):
    """Sao chép bằng gcloud storage, trả về thời gian (giây). Dùng cho cả tải lên và tải về."""
    t = time.time()
    flag = "-r " if recursive else ""
    rc = os.system(f"gcloud storage cp {flag}'{src}' '{dst}' > /dev/null 2>&1")
    if rc != 0:
        raise RuntimeError(f"Không sao chép được {src} -> {dst} (kiểm tra tên bucket / quyền truy cập)")
    return time.time() - t


def dir_size_mb(path):
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs) / 1e6


# =============================================================================
# 2. Chia dữ liệu cho client
# =============================================================================
def iid_split(n, k, seed=42):
    return [np.asarray(p, dtype=int) for p in np.array_split(np.random.default_rng(seed).permutation(n), k)]


def dirichlet_split(y, k, alpha, seed=42):
    """Với mỗi lớp c: q_c ~ Dir(alpha), chia mẫu của lớp c cho k client theo tỉ lệ q_c."""
    rng = np.random.default_rng(seed)
    idx = [[] for _ in range(k)]
    for c in np.unique(y):
        ids = rng.permutation(np.where(y == c)[0])
        cuts = (np.cumsum(rng.dirichlet([alpha] * k)) * len(ids)).astype(int)[:-1]
        for i, p in enumerate(np.split(ids, cuts)):
            idx[i] += p.tolist()
    return [np.asarray(i, dtype=int) for i in idx]


def shard_split(y, k, n_per_client, seed=42):
    """Sắp xếp theo nhãn, cắt k*n mảnh, mỗi client nhận n mảnh (McMahan 2017)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(y, kind="stable")
    shards = np.array_split(order, k * n_per_client)
    perm = rng.permutation(len(shards))
    return [np.concatenate([shards[j] for j in perm[i * n_per_client:(i + 1) * n_per_client]]) for i in range(k)]


def split_indices(y, k, alpha, seed=42):
    """alpha = 0 -> IID; 'shard2' -> 2 mảnh/client; số > 0 -> Dirichlet(alpha)."""
    if isinstance(alpha, str) and alpha.startswith("shard"):
        return shard_split(y, k, int(alpha[5:]), seed)
    if float(alpha) == 0:
        return iid_split(len(y), k, seed)
    return dirichlet_split(y, k, float(alpha), seed)


# =============================================================================
# 3. Dữ liệu ảnh
# =============================================================================
IMG_STATS = {
    "mnist": ([0.1307], [0.3081]),
    "cifar100": ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
}
_raw_cache = {}


def load_image_raw(name, use_gcs=False, bucket=None, root="/content/data"):
    """Trả về x_train, y_train, x_test, y_test (numpy uint8)."""
    if name in _raw_cache:
        return _raw_cache[name]
    if use_gcs:
        p = f"/content/raw_{name}.npz"
        if not os.path.exists(p):
            dt = gcs_cp(f"gs://{bucket}/raw/{name}.npz", p, recursive=False)
            print(f"Tải raw/{name}.npz trong {dt:.1f}s")
        d = np.load(p)
        out = (d["x_train"], d["y_train"], d["x_test"], d["y_test"])
    else:
        import torchvision as tv
        cls = {"mnist": tv.datasets.MNIST, "cifar100": tv.datasets.CIFAR100}[name]
        tr = cls(root, train=True, download=True)
        te = cls(root, train=False, download=True)
        out = (np.asarray(tr.data), np.asarray(tr.targets), np.asarray(te.data), np.asarray(te.targets))
    _raw_cache[name] = out
    return out


def img_to_tensor(x, name):
    x = torch.tensor(x)
    x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, 3, 1, 2)
    x = x.float().div_(255)
    m, s = IMG_STATS[name]
    m = torch.tensor(m).view(1, -1, 1, 1); s = torch.tensor(s).view(1, -1, 1, 1)
    return ((x - m) / s).to(DEV)


def make_image_clients(name, alpha, k, use_gcs=False, bucket=None, seed=42):
    """Trả về (clients, test). clients = [(x, y)], test = (x, y). Tất cả nằm trên GPU.
    Khi use_gcs=True, đọc dữ liệu đã chia sẵn bởi ETL: processed/<name>/alpha<a>_k<k>/client_XXX.npz"""
    xtr, ytr, xte, yte = load_image_raw(name, use_gcs, bucket)
    raw = []
    if use_gcs and not isinstance(alpha, str):
        sub = f"alpha{alpha}_k{k}"
        local = f"/content/processed/{name}/{sub}"
        if not os.path.exists(local):
            os.makedirs(os.path.dirname(local), exist_ok=True)
            dt = gcs_cp(f"gs://{bucket}/processed/{name}/{sub}", os.path.dirname(local))
            print(f"Tải processed/{name}/{sub} trong {dt:.1f}s")
        for i in range(k):
            d = np.load(f"{local}/client_{i:03d}.npz")
            raw.append((d["x"], d["y"]))
    else:
        for idx in split_indices(ytr, k, alpha, seed):
            raw.append((xtr[idx], ytr[idx]))
    clients = [(img_to_tensor(x, name), torch.tensor(y, dtype=torch.long, device=DEV)) for x, y in raw]
    test = (img_to_tensor(xte, name), torch.tensor(yte, dtype=torch.long, device=DEV))
    return clients, test


def augment_cifar(x):
    """Augmentation rẻ trên GPU: lật ngang + dịch ngẫu nhiên ±4 px."""
    if random.random() < 0.5:
        x = x.flip(3)
    dx, dy = random.randint(-4, 4), random.randint(-4, 4)
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


# =============================================================================
# 4. Dữ liệu chuỗi thời gian Milan (Telecom Italia Big Data Challenge)
# =============================================================================
MILAN_COLS = ["cell", "time", "country", "sms_in", "sms_out", "call_in", "call_out", "internet"]


def _milan_colmap(cols):
    """Ánh xạ tên cột (nhiều phiên bản khác nhau trên Kaggle) về tên chuẩn."""
    m = {}
    for c in cols:
        k = c.lower().replace(" ", "").replace("_", "").replace("-", "")
        if k in ("cellid", "squareid", "cell", "gridid", "square"): m[c] = "cell"
        elif k in ("time", "timeinterval", "datetime", "timestamp"): m[c] = "time"
        elif k in ("countrycode", "country"): m[c] = "country"
        elif k in ("smsin",): m[c] = "sms_in"
        elif k in ("smsout",): m[c] = "sms_out"
        elif k in ("callin",): m[c] = "call_in"
        elif k in ("callout",): m[c] = "call_out"
        elif k.startswith("internet"): m[c] = "internet"
    return m


def read_milan_file(path):
    """Đọc 1 file Milan (txt tab, không header – bản gốc Dataverse; hoặc csv có header – bản Kaggle),
    gộp theo (cell, giờ). Trả về DataFrame: cell, hour, sms, call, internet (giờ địa phương Milan)."""
    import pandas as pd
    with open(path, "r", errors="ignore") as f:
        first = f.readline()
    sep = "\t" if "\t" in first else ","
    has_header = any(ch.isalpha() for ch in first.replace("e+", "").replace("E+", "").replace("e-", ""))
    if has_header:
        df = pd.read_csv(path, sep=sep)
        df = df.rename(columns=_milan_colmap(df.columns))
    else:
        df = pd.read_csv(path, sep=sep, header=None)
        df = df.iloc[:, :8]
        df.columns = MILAN_COLS[: df.shape[1]]
    for c in ["sms_in", "sms_out", "call_in", "call_out", "internet"]:
        if c not in df.columns:
            df[c] = 0.0
    df[["sms_in", "sms_out", "call_in", "call_out", "internet"]] = df[["sms_in", "sms_out", "call_in", "call_out", "internet"]].fillna(0.0)
    if pd.api.types.is_numeric_dtype(df["time"]):
        t = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("Europe/Rome").dt.tz_localize(None)
    else:
        t = pd.to_datetime(df["time"])
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_convert("Europe/Rome").dt.tz_localize(None)
    df["hour"] = t.dt.floor("h")
    df["sms"] = df["sms_in"] + df["sms_out"]
    df["call"] = df["call_in"] + df["call_out"]
    out = df.groupby(["cell", "hour"], as_index=False)[["sms", "call", "internet"]].sum()
    out["cell"] = out["cell"].astype(np.int32)
    for c in ["sms", "call", "internet"]:
        out[c] = out[c].astype(np.float32)
    return out


def build_milan_hourly(files, out_path, verbose=True):
    """ETL cục bộ: đọc tất cả file thô -> gộp theo giờ -> lưu 1 file parquet (hoặc csv nếu thiếu pyarrow)."""
    import pandas as pd
    parts, t0, raw_bytes = [], time.time(), 0
    for i, f in enumerate(sorted(files)):
        raw_bytes += os.path.getsize(f)
        parts.append(read_milan_file(f))
        if verbose and (i % 10 == 0 or i == len(files) - 1):
            print(f"  [{i+1}/{len(files)}] {os.path.basename(f)}  ({time.time()-t0:.0f}s)")
    df = pd.concat(parts, ignore_index=True).groupby(["cell", "hour"], as_index=False)[["sms", "call", "internet"]].sum()
    try:
        df.to_parquet(out_path, index=False)
    except Exception:
        out_path = out_path.replace(".parquet", ".csv"); df.to_csv(out_path, index=False)
    info = {"raw_files": len(files), "raw_MB": raw_bytes / 1e6, "rows_hourly": len(df),
            "out_MB": os.path.getsize(out_path) / 1e6, "seconds": time.time() - t0, "out_path": out_path}
    if verbose:
        print(json.dumps(info, indent=1))
    return df, info


def load_milan_matrix(df):
    """DataFrame (cell, hour, sms, call, internet) -> dict ma trận [H, C] + trục thời gian + danh sách cell."""
    import pandas as pd
    hours = pd.date_range(df["hour"].min(), df["hour"].max(), freq="h")
    cells = np.sort(df["cell"].unique())
    out = {"hours": hours, "cells": cells}
    for c in ["internet", "sms", "call"]:
        piv = df.pivot_table(index="hour", columns="cell", values=c, aggfunc="sum")
        out[c] = piv.reindex(index=hours, columns=cells).fillna(0.0).values.astype(np.float32)
    return out


def week_profiles(M):
    """Hồ sơ lưu lượng internet trung bình theo 168 giờ trong tuần, chuẩn hoá theo tổng mỗi cell."""
    hours = M["hours"]
    how = (hours.dayofweek * 24 + hours.hour).values
    X = M["internet"]
    prof = np.stack([X[how == h].mean(0) for h in range(168)], 1)  # [C, 168]
    return prof / (prof.sum(1, keepdims=True) + 1e-9)


def cluster_cells(M, k=3, min_traffic_quantile=0.2, seed=42):
    """Phân cụm cell theo hồ sơ tuần (K-means). Đặt tên cụm dựa trên tỉ lệ
    lưu lượng giờ hành chính ngày thường / buổi tối + cuối tuần."""
    from sklearn.cluster import KMeans
    prof = week_profiles(M)
    total = M["internet"].sum(0)
    keep = total >= np.quantile(total, min_traffic_quantile)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(prof[keep])
    labels = np.full(len(total), -1); labels[keep] = km.labels_
    wk_day = [d * 24 + h for d in range(5) for h in range(9, 18)]
    eve_we = [d * 24 + h for d in range(5) for h in range(19, 24)] + [d * 24 + h for d in (5, 6) for h in range(24)]
    info = []
    for c in range(k):
        p = km.cluster_centers_[c]
        score = p[wk_day].mean() / (p[eve_we].mean() + 1e-12)
        info.append({"cluster": c, "n_cells": int((labels == c).sum()), "office_score": float(score)})
    order = sorted(info, key=lambda d: d["office_score"])
    names = {}
    names[order[0]["cluster"]] = "Dân cư"
    names[order[-1]["cluster"]] = "Văn phòng/Công nghiệp"
    for d in order[1:-1]:
        names[d["cluster"]] = "Hỗn hợp"
    for d in info:
        d["name"] = names[d["cluster"]]
    return labels, info, prof, km.cluster_centers_


def select_clients(M, labels, info, per_cluster=10, names=("Dân cư", "Văn phòng/Công nghiệp"), seed=42):
    """Chọn ngẫu nhiên per_cluster cell trong mỗi cụm có tên thuộc `names`."""
    rng = np.random.default_rng(seed)
    chosen, groups = [], []
    for d in info:
        if d["name"] in names:
            idx = np.where(labels == d["cluster"])[0]
            pick = rng.choice(idx, size=min(per_cluster, len(idx)), replace=False)
            chosen += pick.tolist(); groups += [d["name"]] * len(pick)
    return np.array(chosen), groups


def ts_features(M, ci):
    """Đặc trưng theo giờ cho cell index ci: log1p(internet, sms, call), sin/cos giờ, cờ cuối tuần."""
    h = M["hours"]
    hod = h.hour.values
    feats = np.stack([
        np.log1p(M["internet"][:, ci]), np.log1p(M["sms"][:, ci]), np.log1p(M["call"][:, ci]),
        np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24), (h.dayofweek.values >= 5).astype(np.float32),
    ], 1).astype(np.float32)
    return feats  # [H, 6], cột 0 là mục tiêu (log1p internet)


def windows(feat, L, T):
    """Cửa sổ trượt: X[i] = feat[i:i+L], Y[i] = feat[i+L:i+L+T, 0]."""
    N = len(feat) - L - T + 1
    X = np.stack([feat[i:i + L] for i in range(N)])
    Y = np.stack([feat[i + L:i + L + T, 0] for i in range(N)])
    return X, Y


def make_milan_clients(M, cell_idx, groups=None, L=24, T=1, split=(0.7, 0.15)):
    """Mỗi client = 1 cell. Chia theo thời gian (không xáo trộn): 70% train, 15% val, 15% test.
    Chuẩn hoá 3 cột lưu lượng theo mean/std của tập train của chính client đó (client tự làm tại Edge)."""
    clients = []
    for j, ci in enumerate(cell_idx):
        f = ts_features(M, ci)
        H = len(f); a = int(H * split[0]); b = int(H * (split[0] + split[1]))
        mu = f[:a, :3].mean(0); sd = f[:a, :3].std(0) + 1e-6
        g = f.copy(); g[:, :3] = (g[:, :3] - mu) / sd
        Xtr, Ytr = windows(g[:a], L, T)
        Xva, Yva = windows(g[a - L:b], L, T)
        Xte, Yte = windows(g[b - L:], L, T)
        t = lambda z: torch.tensor(z, dtype=torch.float32, device=DEV)
        clients.append({"cell": int(M["cells"][ci]), "group": groups[j] if groups else "",
                        "train": (t(Xtr), t(Ytr)), "val": (t(Xva), t(Yva)), "test": (t(Xte), t(Yte)),
                        "mu": float(mu[0]), "sd": float(sd[0]), "raw_hourly_bytes": int(f.nbytes)})
    return clients


# =============================================================================
# 5. Mô hình
# =============================================================================
class CNN_MNIST(nn.Module):
    """Kiến trúc CNN của McMahan et al. (2017)."""
    def __init__(self, n_cls=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 5, padding=2); self.c2 = nn.Conv2d(32, 64, 5, padding=2)
        self.f1 = nn.Linear(64 * 7 * 7, 512); self.f2 = nn.Linear(512, n_cls)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.f2(F.relu(self.f1(x.flatten(1))))


def _blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(inplace=True))


class CNN_CIFAR(nn.Module):
    """CNN 6 lớp conv + GroupNorm (BatchNorm kém trong FL Non-IID – Hsieh et al. 2020)."""
    def __init__(self, n_cls=100):
        super().__init__()
        self.f = nn.Sequential(_blk(3, 64), _blk(64, 64), nn.MaxPool2d(2),
                               _blk(64, 128), _blk(128, 128), nn.MaxPool2d(2),
                               _blk(128, 256), _blk(256, 256), nn.MaxPool2d(2))
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256 * 16, 512),
                               nn.ReLU(inplace=True), nn.Linear(512, n_cls))

    def forward(self, x):
        return self.h(self.f(x))


class RNNForecaster(nn.Module):
    """GRU hoặc LSTM dự báo T giờ tới từ L giờ quá khứ."""
    def __init__(self, n_feat=6, hidden=64, layers=2, T=1, cell="gru", dropout=0.1):
        super().__init__()
        R = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = R(n_feat, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, T)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# 6. Task: định nghĩa loss + cách đánh giá cho phân loại / hồi quy
# =============================================================================
class ClsTask:
    """Phân loại ảnh. eval_data = (x_test, y_test). Chỉ số chính: accuracy (càng cao càng tốt)."""
    main, higher_better = "acc", True

    def __init__(self, augment=None):
        self.augment = augment

    def loss(self, out, y):
        return F.cross_entropy(out, y)

    @torch.no_grad()
    def evaluate(self, model, eval_data, bs=2000):
        model.eval(); x, y = eval_data; correct, loss = 0, 0.0
        for i in range(0, len(y), bs):
            o = model(x[i:i + bs])
            loss += F.cross_entropy(o, y[i:i + bs], reduction="sum").item()
            correct += (o.argmax(1) == y[i:i + bs]).sum().item()
        return {"acc": correct / len(y), "loss": loss / len(y)}


class RegTask:
    """Dự báo chuỗi thời gian. eval_data = list client dict (dùng client['test'] hoặc ['val']).
    Chỉ số chính: RMSE trên giá trị đã chuẩn hoá (càng thấp càng tốt).
    Thêm MAE theo đơn vị gốc của dữ liệu Milan (đảo chuẩn hoá + expm1)."""
    main, higher_better = "rmse", False
    augment = None

    def __init__(self, split="test"):
        self.split = split

    def loss(self, out, y):
        return F.mse_loss(out, y)

    @torch.no_grad()
    def evaluate(self, model, eval_data, bs=4096, per_client=False):
        model.eval(); se, n, ae_orig = 0.0, 0, 0.0; per = []
        for c in eval_data:
            x, y = c[self.split]
            p = torch.cat([model(x[i:i + bs]) for i in range(0, len(y), bs)])
            e = ((p - y) ** 2).sum().item()
            se += e; n += y.numel()
            orig_p = torch.expm1(p * c["sd"] + c["mu"]).clamp_min(0)
            orig_y = torch.expm1(y * c["sd"] + c["mu"])
            ae_orig += (orig_p - orig_y).abs().sum().item()
            if per_client:
                per.append({"cell": c["cell"], "group": c["group"], "rmse": math.sqrt(e / y.numel())})
        res = {"rmse": math.sqrt(se / n), "mae_orig": ae_orig / n, "loss": se / n}
        if per_client:
            res["per_client"] = per
        return res


def _client_xy(c):
    return c["train"] if isinstance(c, dict) else c


# =============================================================================
# 7. Nén cập nhật mô hình
# =============================================================================
def compress(vec, method="none"):
    """Nén vector cập nhật. Trả về (vector sau giải nén, số byte truyền thực tế).
    method: none | fp16 | int8 | int4 | topk:<p> | topk_int8:<p>   (p = tỉ lệ phần tử giữ lại, ví dụ 0.01)"""
    n = vec.numel()
    if method == "none":
        return vec, 4 * n
    if method == "fp16":
        return vec.half().float(), 2 * n
    if method in ("int8", "int4", "int2"):
        b = int(method[3:]); qmax = 2 ** (b - 1) - 1
        s = vec.abs().max().clamp_min(1e-12) / qmax
        q = torch.round(vec / s).clamp(-qmax, qmax)
        return q * s, math.ceil(n * b / 8) + 4
    if method.startswith("topk"):
        kind, p = method.split(":"); p = float(p)
        k = max(1, int(n * p))
        idx = vec.abs().topk(k).indices
        vals = vec[idx]
        if kind == "topk_int8":
            s = vals.abs().max().clamp_min(1e-12) / 127
            vals = torch.round(vals / s).clamp(-127, 127) * s
            nbytes = k * (1 + 4) + 4          # giá trị int8 + chỉ số int32 + scale
        else:
            nbytes = k * (4 + 4)              # giá trị fp32 + chỉ số int32
        out = torch.zeros_like(vec); out[idx] = vals
        return out, nbytes
    raise ValueError(f"Phương pháp nén không hợp lệ: {method}")


def get_vec(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()


def set_vec(model, vec):
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


# =============================================================================
# 8. Vòng lặp FL tổng quát
# =============================================================================
def _make_opt(model, name, lr):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    return torch.optim.SGD(model.parameters(), lr=lr)


def local_train(task, model, data, epochs, bs, lr, mu=0.0, opt="sgd", clip=None, optimizer=None):
    """Huấn luyện cục bộ (FedAvg nếu mu=0, FedProx nếu mu>0). Trả về số bước đã chạy."""
    x, y = data; n = len(y)
    if n == 0:
        return 0
    model.train()
    w0 = [p.detach().clone() for p in model.parameters()]
    o = optimizer or _make_opt(model, opt, lr); steps = 0
    for _ in range(epochs):
        perm = torch.randperm(n, device=x.device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            xb = x[b] if task.augment is None else task.augment(x[b])
            loss = task.loss(model(xb), y[b])
            if mu > 0:
                loss = loss + mu / 2 * sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), w0))
            o.zero_grad(set_to_none=True); loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            o.step(); steps += 1
    return steps


def full_gradient(task, model, data, chunk=1000):
    """FedSGD: gradient trên toàn bộ dữ liệu client (cộng dồn theo chunk)."""
    x, y = data; n = len(y)
    model.train(); model.zero_grad(set_to_none=True)
    for i in range(0, n, chunk):
        out = model(x[i:i + chunk])
        l = task.loss(out, y[i:i + chunk]) * (len(y[i:i + chunk]) / n)
        l.backward()
    return torch.cat([p.grad.detach().flatten() for p in model.parameters()])


def run_fl(task, model_fn, clients, eval_data, algo="fedavg", rounds=100, frac=0.1, E=5, B=50,
           lr=0.01, mu=0.0, opt="sgd", clip=None, compress_method="none", error_feedback=True,
           straggler=0.0, server_lr=None, eval_every=1, target=None, tag="", seed=42,
           results_dir=None, save_ckpt_every=0, verbose=True):
    """Chạy 1 thí nghiệm FL.
    algo: fedsgd | fedavg | fedprox.  Client gửi Δw (FedAvg/FedProx) hoặc gradient (FedSGD), có thể nén.
    Không nén thì kết quả tương đương công thức FedAvg gốc: w^{t+1} = Σ (n_k/n) w_k^{t+1}."""
    set_seed(seed)
    gmodel = model_fn().to(DEV)
    w = get_vec(gmodel); P = w.numel()
    K = len(clients); m = max(1, int(round(frac * K)))
    sizes = [len(_client_xy(c)[1]) for c in clients]
    residual = {}                                       # error feedback cho từng client
    log = {"tag": tag, "algo": algo, "K": K, "m": m, "frac": frac, "E": E, "B": B, "lr": lr, "mu": mu,
           "opt": opt, "compress": compress_method, "straggler": straggler, "n_params": P,
           "round": [], "time": [], "bytes_up_cum": [], "bytes_down_cum": [], task.main: [], "metrics": []}
    up, down = 0, 0; t0 = time.time(); reached = None
    work = copy.deepcopy(gmodel)
    for r in range(1, rounds + 1):
        sel = [i for i in np.random.choice(K, m, replace=False) if sizes[i] > 0]
        tot = float(sum(sizes[i] for i in sel))
        agg = torch.zeros_like(w)
        for i in sel:
            set_vec(work, w.clone()); down += 4 * P          # server phát mô hình fp32 xuống
            data = _client_xy(clients[i])
            if algo == "fedsgd":
                upd = full_gradient(task, work, data)
            else:
                ep = E
                if straggler > 0 and E > 1 and random.random() < straggler:
                    ep = random.randint(1, E - 1)
                local_train(task, work, data, ep, B, lr, mu if algo == "fedprox" else 0.0, opt, clip)
                upd = get_vec(work) - w                      # Δw
            if compress_method != "none":
                if error_feedback:
                    upd = upd + residual.get(i, 0)
                sent, nb = compress(upd, compress_method)
                if error_feedback:
                    residual[i] = upd - sent
            else:
                sent, nb = upd, 4 * P
            up += nb
            agg += (sizes[i] / tot) * sent
        w = w - (server_lr or lr) * agg if algo == "fedsgd" else w + agg
        if r % eval_every == 0 or r == rounds:
            set_vec(gmodel, w)
            res = task.evaluate(gmodel, eval_data)
            log["round"].append(r); log["time"].append(time.time() - t0)
            log["bytes_up_cum"].append(up); log["bytes_down_cum"].append(down)
            log[task.main].append(res[task.main]); log["metrics"].append(res)
            if target is not None and reached is None:
                ok = res[task.main] >= target if task.higher_better else res[task.main] <= target
                if ok: reached = r
            if verbose and (r % max(1, rounds // 10) == 0 or r == rounds):
                print(f"[{tag}] round {r:4d}  {task.main}={res[task.main]:.4f}  "
                      f"up={up/1e6:.2f}MB  {time.time()-t0:.0f}s")
        if save_ckpt_every and results_dir and r % save_ckpt_every == 0:
            d = f"{results_dir}/ckpt/{tag}"; os.makedirs(d, exist_ok=True)
            set_vec(gmodel, w); torch.save(gmodel.state_dict(), f"{d}/round_{r:04d}.pt")
    set_vec(gmodel, w)
    vals = log[task.main]
    log.update({"final": vals[-1], "best": max(vals) if task.higher_better else min(vals),
                "rounds_to_target": reached, "target": target, "total_time_s": time.time() - t0,
                "bytes_up": up, "bytes_down": down})
    if hasattr(task, "split") and isinstance(eval_data, list):
        log["final_per_client"] = task.evaluate(gmodel, eval_data, per_client=True)["per_client"]
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        with open(f"{results_dir}/{tag}.json", "w") as f:
            json.dump(log, f)
    return log, gmodel


# =============================================================================
# 9. Baseline
# =============================================================================
def train_centralized(task, model_fn, clients, eval_data, epochs=20, B=64, lr=1e-3, opt="adam", clip=None,
                      tag="centralized", seed=42, results_dir=None, verbose=True):
    """Gom toàn bộ dữ liệu train của mọi client về 1 chỗ rồi huấn luyện (học tập trung)."""
    set_seed(seed)
    xs = torch.cat([_client_xy(c)[0] for c in clients]); ys = torch.cat([_client_xy(c)[1] for c in clients])
    model = model_fn().to(DEV); log = {"tag": tag, "epoch": [], task.main: [], "metrics": []}
    data_bytes = xs.numel() * 4 + ys.numel() * 4
    t0 = time.time(); o = _make_opt(model, opt, lr)
    for e in range(1, epochs + 1):
        local_train(task, model, (xs, ys), 1, B, lr, 0.0, opt, clip, optimizer=o)
        res = task.evaluate(model, eval_data)
        log["epoch"].append(e); log[task.main].append(res[task.main]); log["metrics"].append(res)
        if verbose and (e % max(1, epochs // 5) == 0 or e == epochs):
            print(f"[{tag}] epoch {e:3d}  {task.main}={res[task.main]:.4f}")
    vals = log[task.main]
    log.update({"final": vals[-1], "best": max(vals) if task.higher_better else min(vals),
                "data_bytes_window_tensors": data_bytes, "total_time_s": time.time() - t0})
    if hasattr(task, "split") and isinstance(eval_data, list):
        log["final_per_client"] = task.evaluate(model, eval_data, per_client=True)["per_client"]
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        json.dump(log, open(f"{results_dir}/{tag}.json", "w"))
    return log, model


def train_local_only(task, model_fn, clients, epochs=20, B=64, lr=1e-3, opt="adam", clip=None, seed=42):
    """Mỗi client tự huấn luyện mô hình riêng, không chia sẻ gì (chỉ cho RegTask)."""
    set_seed(seed); per = []; se, n = 0.0, 0
    for c in clients:
        m = model_fn().to(DEV)
        local_train(task, m, c["train"], epochs, B, lr, 0.0, opt, clip)
        r = task.evaluate(m, [c], per_client=True)
        nn_ = c["test"][1].numel(); se += r["rmse"] ** 2 * nn_; n += nn_
        per.append(r["per_client"][0])
    return {"tag": "local_only", "final": math.sqrt(se / n), "final_per_client": per}


def naive_seasonal(clients, split="test"):
    """Baseline ngây thơ: dự báo = giá trị cùng giờ của 24 giờ trước (cần L = 24)."""
    se, n, per = 0.0, 0, []
    for c in clients:
        x, y = c[split]; T = y.shape[1]
        pred = x[:, :T, 0]                     # x[:, j, 0] là giá trị lúc (t+1+j) − 24
        e = ((pred - y) ** 2).sum().item(); se += e; n += y.numel()
        per.append({"cell": c["cell"], "group": c["group"], "rmse": math.sqrt(e / y.numel())})
    return {"tag": "naive_24h", "final": math.sqrt(se / n), "final_per_client": per}


# =============================================================================
# 10. Đo độ trễ Edge vs Cloud
# =============================================================================
@torch.no_grad()
def measure_inference_ms(model, x_one, device="cpu", threads=None, reps=200):
    """Thời gian suy luận 1 mẫu (median, ms). threads=1 để giả lập thiết bị Edge yếu."""
    old = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    m = copy.deepcopy(model).to(device).eval(); x = x_one.to(device)
    for _ in range(20):
        m(x)
    if device == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); m(x)
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    torch.set_num_threads(old)
    return float(np.median(ts))


def measure_gcs_rtt_ms(buckets, reps=20, obj="bench/ping.txt"):
    """Đo thời gian khứ hồi (RTT) tới bucket ở từng region: tải 1 object rất nhỏ nhiều lần, lấy median.
    buckets: dict {"Singapore": "fl-sem-vinh-sg", ...}"""
    from google.cloud import storage
    cli = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT")); out = {}
    for region, name in buckets.items():
        blob = cli.bucket(name).blob(obj)
        blob.upload_from_string("ping")
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); blob.download_as_bytes(); ts.append((time.perf_counter() - t) * 1000)
        out[region] = float(np.median(ts))
    return out


def cloud_latency_ms(rtt_ms, payload_bytes, uplink_mbps, infer_ms):
    """Độ trễ khi gửi dữ liệu về Cloud dự đoán = RTT + thời gian truyền payload + suy luận trên Cloud."""
    return rtt_ms + payload_bytes * 8 / (uplink_mbps * 1e6) * 1000 + infer_ms


# =============================================================================
# 11. Vẽ biểu đồ và bảng
# =============================================================================
def plot_runs(logs, title, fname=None, metric=None, x="round", ylabel=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.3))
    for L in logs:
        mkey = metric or ("acc" if "acc" in L else "rmse")
        xs = L[x] if x in L else L.get("round", L.get("epoch"))
        if x == "bytes_up_cum":
            xs = np.array(xs) / 1e6
        plt.plot(xs, L[mkey], label=L["tag"], lw=1.6)
    plt.xlabel({"round": "Vòng giao tiếp", "bytes_up_cum": "Dung lượng client gửi lên (MB)"}.get(x, x))
    plt.ylabel(ylabel or (metric or "metric")); plt.title(title); plt.grid(alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
    plt.show()


def table(logs, main=None):
    rows = []
    head = f"{'Thí nghiệm':42s} {'Final':>8s} {'Best':>8s} {'R→target':>9s} {'Up MB':>9s} {'Down MB':>9s} {'Time s':>7s}"
    print(head); print("-" * len(head))
    for L in logs:
        rt = L.get("rounds_to_target") or "-"
        print(f"{L['tag']:42s} {L['final']:8.4f} {L.get('best', L['final']):8.4f} {str(rt):>9s} "
              f"{L.get('bytes_up', 0)/1e6:9.2f} {L.get('bytes_down', 0)/1e6:9.2f} {L.get('total_time_s', 0):7.0f}")
        rows.append({k: L.get(k) for k in ["tag", "final", "best", "rounds_to_target", "bytes_up", "bytes_down", "total_time_s"]})
    return rows
