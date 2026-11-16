"""Vẽ 2 hình phân cụm cho báo cáo + in cấu hình máy (CPU, RAM).

Chạy trong thư mục flower_5g (cùng chỗ với prepare_data.py):
    python make_cluster_figs.py

Đầu vào : ~/fl5g_data/milan_hourly.parquet  (prepare_data.py đã tạo)
Đầu ra  : ~/fl5g_results/clusters_profile.png, ~/fl5g_results/clusters_map.png
Dùng đúng hàm cluster_cells (seed 42) như prepare_data.py nên kết quả trùng Bảng 2.3.
"""
import os, platform, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from fl5g import flcore as fc

DATA = Path(os.environ.get("FL5G_DATA", Path.home() / "fl5g_data"))
OUT = Path(os.environ.get("FL5G_OUT", Path.home() / "fl5g_results"))
OUT.mkdir(parents=True, exist_ok=True)

# ---------- cấu hình máy ----------
def ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 2**30
    except Exception:
        pass
    if platform.system() == "Windows":
        import ctypes
        class M(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = M(); m.dwLength = ctypes.sizeof(M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 2**30
    return float("nan")

cpu = platform.processor()
if platform.system() == "Windows":
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        pass
print(f"CPU : {cpu}  ({os.cpu_count()} luồng)")
print(f"RAM : {ram_gb():.1f} GB")
print(f"OS  : {platform.system()} {platform.release()}  |  Python {platform.python_version()}")

# ---------- phân cụm ----------
df = pd.read_parquet(DATA / "milan_hourly.parquet")
M = fc.load_milan_matrix(df)
labels, info, prof, centers = fc.cluster_cells(M, k=3)
idx, groups = fc.select_clients(M, labels, info, per_cluster=10)
print(json.dumps(info, ensure_ascii=False, indent=1))

COL = {"Dân cư": "#eb6834", "Văn phòng/Công nghiệp": "#2a78d6", "Hỗn hợp": "#8a8f98"}
name = {d["cluster"]: d["name"] for d in info}
ncell = {d["cluster"]: d["n_cells"] for d in info}

# Hình 2.2 – hồ sơ tuần (168 giờ) của tâm mỗi cụm
fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
for c in sorted(name, key=lambda c: name[c]):
    ax.plot(np.arange(168), centers[c] * 168, lw=2, color=COL[name[c]], label=f"{name[c]} ({ncell[c]:,} ô)".replace(",", "."))
days = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
ax.set_xticks([d * 24 + 12 for d in range(7)]); ax.set_xticklabels(days)
for d in range(1, 7): ax.axvline(d * 24, color="#cccccc", lw=0.8)
ax.axvspan(120, 168, color="#f2f2f2", zorder=0)
ax.set_xlim(0, 167); ax.set_xlabel("Giờ trong tuần"); ax.set_ylabel("Lưu lượng internet tương đối\n(1 = mức trung bình)")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
plt.tight_layout(); plt.savefig(OUT / "clusters_profile.png"); plt.close()

# Hình 2.3 – bản đồ cụm trên lưới 100 × 100 + 20 trạm được chọn
grid = np.full(10000, np.nan)
code = {"Dân cư": 0, "Hỗn hợp": 1, "Văn phòng/Công nghiệp": 2}
for i, cell in enumerate(M["cells"]):
    if labels[i] >= 0: grid[int(cell) - 1] = code[name[labels[i]]]
grid = grid.reshape(100, 100)
cmap = ListedColormap([COL["Dân cư"], COL["Hỗn hợp"], COL["Văn phòng/Công nghiệp"]]); cmap.set_bad("#ffffff")
fig, ax = plt.subplots(figsize=(6.4, 6), dpi=200)
ax.imshow(grid, cmap=cmap, origin="lower", interpolation="nearest", vmin=0, vmax=2)
for i, g in zip(idx, groups):
    cell = int(M["cells"][i]) - 1
    ax.scatter(cell % 100, cell // 100, s=60, facecolor="none", edgecolor="#111111", lw=1.6)
from matplotlib.patches import Patch
h = [Patch(color=COL[n], label=n) for n in ["Dân cư", "Hỗn hợp", "Văn phòng/Công nghiệp"]]
h.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#111111", label="20 trạm được chọn"))
h.append(Patch(facecolor="#ffffff", edgecolor="#999999", label="Lưu lượng thấp (loại)"))
ax.legend(handles=h, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Lưới 100 × 100 ô của Milan (mỗi ô ≈ 235 m)", fontsize=10)
plt.tight_layout(); plt.savefig(OUT / "clusters_map.png"); plt.close()
print(f"Đã lưu {OUT / 'clusters_profile.png'} và {OUT / 'clusters_map.png'}")
