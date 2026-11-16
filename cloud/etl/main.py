"""ETL tự động (Cloud Run functions – FaaS).
Trigger: mỗi khi có file mới được upload vào  gs://<bucket>/raw/milan/*.txt|csv
  → đọc file, làm sạch, gộp theo (ô lưới, giờ) → ghi  processed/milan/hourly/<tên>.parquet
  → ghi log thời gian xử lý vào  processed/milan/etl_log/<tên>.json
"""
import functions_framework, json, os, time
import numpy as np
import pandas as pd
from google.cloud import storage

cli = storage.Client()

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




def milan_file(b, name):
    t0 = time.time()
    local = "/tmp/" + os.path.basename(name)
    b.blob(name).download_to_filename(local)
    t_dl = time.time() - t0
    df = read_milan_file(local)
    base = os.path.splitext(os.path.basename(name))[0]
    out = f"/tmp/{base}.parquet"; df.to_parquet(out, index=False)
    b.blob(f"processed/milan/hourly/{base}.parquet").upload_from_filename(out)
    log = {"file": name, "raw_MB": os.path.getsize(local) / 1e6, "out_MB": os.path.getsize(out) / 1e6,
           "rows_out": len(df), "download_s": t_dl, "total_s": time.time() - t0}
    b.blob(f"processed/milan/etl_log/{base}.json").upload_from_string(json.dumps(log))
    os.remove(local); os.remove(out)
    return log



@functions_framework.cloud_event
def etl(event):
    name, bkt = event.data["name"], event.data["bucket"]
    if name.startswith("raw/milan/") and name.endswith((".txt", ".csv")):
        print("ETL Milan xong:", json.dumps(milan_file(cli.bucket(bkt), name)))
    # các file khác (processed/, results/ ...) bỏ qua để tránh tự kích hoạt lặp
