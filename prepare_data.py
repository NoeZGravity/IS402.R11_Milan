"""Chuẩn bị dữ liệu cho Flower: file thô Milan -> gộp theo giờ -> phân cụm -> chọn trạm -> cửa sổ trượt.

Ví dụ:
  python prepare_data.py --raw "D:/data/milan_raw"                 # từ các file .txt tải từ Harvard
  python prepare_data.py --hourly "D:/data/milan_hourly.parquet"   # nếu đã có file gộp theo giờ (từ Colab)
Kết quả lưu ở ~/fl5g_data (đổi bằng biến môi trường FL5G_DATA).
"""
import argparse, glob, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from fl5g import flcore as fc

ap = argparse.ArgumentParser()
ap.add_argument("--raw", help="thư mục chứa file .txt/.csv thô")
ap.add_argument("--hourly", help="file parquet/csv đã gộp theo giờ")
ap.add_argument("--max-files", type=int, default=None)
ap.add_argument("--per-cluster", type=int, default=10, help="số trạm mỗi khu vực")
ap.add_argument("--L", type=int, default=24); ap.add_argument("--T", type=int, default=1)
ap.add_argument("--out", default=os.environ.get("FL5G_DATA", str(Path.home() / "fl5g_data")))
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

t0 = time.time(); etl = {}
if a.hourly:
    df = pd.read_parquet(a.hourly) if a.hourly.endswith(".parquet") else pd.read_csv(a.hourly, parse_dates=["hour"])
else:
    files = sorted(f for f in glob.glob(os.path.join(a.raw, "**", "*"), recursive=True) if f.endswith((".txt", ".csv")))
    files = files[: a.max_files] if a.max_files else files
    assert files, "Không tìm thấy file .txt/.csv trong --raw"
    df, etl = fc.build_milan_hourly(files, str(out / "milan_hourly.parquet"))
M = fc.load_milan_matrix(df)
labels, cinfo, _, _ = fc.cluster_cells(M, k=3)
for d in cinfo: print(d)
idx, groups = fc.select_clients(M, labels, cinfo, per_cluster=a.per_cluster)
clients = fc.make_milan_clients(M, idx, groups, L=a.L, T=a.T)

arrays, meta = {}, {"T": a.T, "L": a.L, "clients": [], "clusters": cinfo, "etl": etl,
                    "hours": [str(M["hours"][0]), str(M["hours"][-1])], "n_cells_total": int(len(M["cells"]))}
for i, c in enumerate(clients):
    for k in ("train", "val", "test"):
        arrays[f"{k}_X_{i}"] = c[k][0].cpu().numpy(); arrays[f"{k}_Y_{i}"] = c[k][1].cpu().numpy()
    meta["clients"].append({"cell": c["cell"], "group": c["group"], "mu": c["mu"], "sd": c["sd"],
                            "raw_hourly_bytes": c["raw_hourly_bytes"], "n_train": int(len(c["train"][1]))})
np.savez_compressed(out / "clients.npz", **arrays)
json.dump(meta, open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(f"\nXong trong {time.time()-t0:.0f}s: {len(clients)} trạm -> {out}")
print("Chạy FL với số SuperNode = số trạm:")
print(f'  flwr run . --stream --federation-config "num-supernodes={len(clients)}"')
