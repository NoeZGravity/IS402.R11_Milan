"""Chép mô hình + dữ liệu đã chuẩn bị vào thư mục api/ để chạy API / đóng gói Docker.
   python prepare_api.py                     (mặc định dùng kết quả fedavg_none)
   python prepare_api.py --tag fedavg_int8   (dùng mô hình của thí nghiệm khác)"""
import argparse, glob, json, os, shutil
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="fedavg_none"); a = ap.parse_args()
DATA = Path(os.environ.get("FL5G_DATA", Path.home() / "fl5g_data"))
OUT = Path(os.environ.get("FL5G_OUT", Path.home() / "fl5g_results"))
api = Path(__file__).parent / "api"
(api / "data" / "results").mkdir(parents=True, exist_ok=True); (api / "model").mkdir(exist_ok=True)
shutil.copy(DATA / "meta.json", api / "data" / "meta.json")
shutil.copy(DATA / "clients.npz", api / "data" / "clients.npz")
res = json.load(open(OUT / f"{a.tag}.json", encoding="utf-8"))
shutil.copy(OUT / f"{a.tag}_model.pt", api / "model" / "model.pt")
cfg = res["config"]
json.dump({"tag": a.tag, "cell": cfg.get("cell", "gru"), "hidden": cfg.get("hidden", 64), "layers": cfg.get("layers", 2)},
          open(api / "model" / "model_config.json", "w"), indent=1)
for f in glob.glob(str(OUT / "*.json")):
    shutil.copy(f, api / "data" / "results" / Path(f).name)
size = sum(p.stat().st_size for p in api.rglob("*") if p.is_file()) / 1e6
print(f"Đã chuẩn bị api/ với mô hình {a.tag} ({size:.1f} MB). Chạy thử:  cd api && uvicorn main:app --port 8080")
