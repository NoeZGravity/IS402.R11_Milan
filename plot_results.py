"""Vẽ biểu đồ + bảng từ ~/fl5g_results/*.json (đổi bằng FL5G_OUT)."""
import glob, json, os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
OUT = Path(os.environ.get("FL5G_OUT", Path.home() / "fl5g_results"))
runs = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(str(OUT / "*.json")))]
rows = []
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for r in runs:
    ev = [e for e in r["eval"] if e["round"] > 0]
    up = 0; xs = []
    for e in ev:
        up += next((x["uplink_bytes"] for x in r["rounds"] if x["round"] == e["round"]), 0); xs.append(up / 1e6)
    ax[0].plot([e["round"] for e in ev], [e["rmse"] for e in ev], label=r["tag"])
    ax[1].plot(xs, [e["rmse"] for e in ev], label=r["tag"])
    last = r["eval"][-1]
    rows.append({"Cấu hình": r["tag"], "RMSE": last["rmse"], "RMSE dân cư": last.get("rmse_dan_cu"),
                 "RMSE văn phòng": last.get("rmse_van_phong"), "Uplink (MB)": r["total_uplink_MB"],
                 "Downlink (MB)": r["total_downlink_MB"]})
ax[0].set_xlabel("Vòng giao tiếp"); ax[0].set_ylabel("RMSE (chuẩn hoá)"); ax[0].set_title("Hội tụ")
ax[1].set_xlabel("Tổng dung lượng gửi lên (MB)"); ax[1].set_ylabel("RMSE (chuẩn hoá)"); ax[1].set_title("Độ chính xác theo chi phí truyền")
for a_ in ax: a_.grid(alpha=0.3); a_.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "flower_results.png", dpi=150)
tbl = pd.DataFrame(rows)
if len(tbl):
    base = tbl.loc[tbl["Cấu hình"] == "fedavg_none", "Uplink (MB)"]
    if len(base): tbl["Tỉ lệ nén"] = base.iloc[0] / tbl["Uplink (MB)"]
print(tbl.round(4).to_string(index=False)); tbl.to_csv(OUT / "flower_results.csv", index=False)
print("Đã lưu", OUT / "flower_results.png")
