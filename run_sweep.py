"""Chạy lần lượt nhiều cấu hình (Bước 3: Accuracy vs Communication + Non-IID FedProx).
   python run_sweep.py            (đọc số trạm từ ~/fl5g_data/meta.json)"""
import json, os, subprocess
from pathlib import Path
meta = json.load(open(Path(os.environ.get("FL5G_DATA", Path.home() / "fl5g_data")) / "meta.json", encoding="utf-8"))
n = len(meta["clients"])
runs = [f"compress='{m}'" for m in ["none", "fp16", "int8", "int4", "topk:0.1", "topk:0.01", "topk_int8:0.01"]]
runs.append("algorithm='fedprox' compress='none'")
for rc in runs:
    print("\n==========", rc)
    subprocess.run(["flwr", "run", ".", "--stream", "--run-config", rc,
                    "--federation-config", f"num-supernodes={n}"], check=False)
