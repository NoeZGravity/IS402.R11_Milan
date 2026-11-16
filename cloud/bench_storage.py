"""Đo tốc độ đọc/ghi Cloud Storage và độ trễ (RTT) tới 3 region  – tiêu chí 3.1, 3.3, 3.4.
Chạy trong Cloud Shell (hoặc máy có gcloud auth application-default login):
   pip install google-cloud-storage pandas pyarrow
   python cloud/bench_storage.py --prefix fl5g-vinh --raw sms-call-internet-mi-2013-11-01.txt
Kết quả: bench_results.csv (dán vào báo cáo).
"""
import argparse, io, os, statistics, time
import numpy as np, pandas as pd
from google.cloud import storage

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", required=True); ap.add_argument("--raw", help="1 file .txt Milan để so sánh txt vs Parquet")
ap.add_argument("--mb", type=int, default=100, help="kích thước file thử (MB)"); ap.add_argument("--reps", type=int, default=15)
a = ap.parse_args()
cli = storage.Client()
BUCKETS = {"Singapore (asia-southeast1)": f"{a.prefix}-sg", "US (us-central1)": f"{a.prefix}-us", "EU (europe-west1)": f"{a.prefix}-eu"}
rows = []
def rec(test, region, value, unit): rows.append({"Phép đo": test, "Region": region, "Giá trị": round(value, 2), "Đơn vị": unit}); print(rows[-1])

blob_data = os.urandom(a.mb * 1024 * 1024)
for region, name in BUCKETS.items():
    b = cli.bucket(name)
    # 1) RTT: tải object rất nhỏ nhiều lần
    b.blob("bench/ping.txt").upload_from_string("ping")
    t = []
    for _ in range(a.reps):
        s = time.perf_counter(); b.blob("bench/ping.txt").download_as_bytes(); t.append((time.perf_counter() - s) * 1000)
    rec("RTT (object 4 byte, trung vị)", region, statistics.median(t), "ms")
    # 2) Ghi / đọc file lớn
    s = time.perf_counter(); b.blob("bench/big.bin").upload_from_string(blob_data); up = time.perf_counter() - s
    s = time.perf_counter(); b.blob("bench/big.bin").download_as_bytes(); dn = time.perf_counter() - s
    rec(f"Ghi file {a.mb} MB", region, a.mb / up, "MB/s"); rec(f"Đọc file {a.mb} MB", region, a.mb / dn, "MB/s")

# 3) Tối ưu: nhiều file nhỏ vs 1 file gộp (cùng tổng dung lượng 10 MB) – bucket chính
b = cli.bucket(BUCKETS["Singapore (asia-southeast1)"]); small = os.urandom(100 * 1024)
s = time.perf_counter()
for i in range(100): b.blob(f"bench/small/{i:03d}.bin").upload_from_string(small)
rec("Ghi 100 file × 100 KB", "Singapore", time.perf_counter() - s, "giây")
s = time.perf_counter(); b.blob("bench/merged.bin").upload_from_string(small * 100)
rec("Ghi 1 file gộp 10 MB", "Singapore", time.perf_counter() - s, "giây")

# 4) Tối ưu định dạng: txt thô vs Parquet (đọc + parse 1 ngày dữ liệu)
if a.raw:
    raw_bytes = open(a.raw, "rb").read()
    b.blob("bench/day.txt").upload_from_string(raw_bytes)
    cols = ["cell", "time", "country", "sms_in", "sms_out", "call_in", "call_out", "internet"]
    df = pd.read_csv(a.raw, sep="\t", header=None, names=cols).fillna(0)
    df["hour"] = pd.to_datetime(df["time"], unit="ms").dt.floor("h")
    df = df.groupby(["cell", "hour"], as_index=False)[cols[3:]].sum()
    buf = io.BytesIO(); df.to_parquet(buf, index=False)
    b.blob("bench/day.parquet").upload_from_string(buf.getvalue())
    rec("Dung lượng 1 ngày – txt thô", "-", len(raw_bytes) / 1e6, "MB"); rec("Dung lượng 1 ngày – Parquet (đã gộp giờ)", "-", buf.tell() / 1e6, "MB")
    s = time.perf_counter(); data = b.blob("bench/day.txt").download_as_bytes()
    pd.read_csv(io.BytesIO(data), sep="\t", header=None); rec("Đọc + parse txt từ GCS", "Singapore", time.perf_counter() - s, "giây")
    s = time.perf_counter(); pd.read_parquet(io.BytesIO(b.blob("bench/day.parquet").download_as_bytes()))
    rec("Đọc + parse Parquet từ GCS", "Singapore", time.perf_counter() - s, "giây")

pd.DataFrame(rows).to_csv("bench_results.csv", index=False, encoding="utf-8-sig")
print("\nĐã lưu bench_results.csv")
