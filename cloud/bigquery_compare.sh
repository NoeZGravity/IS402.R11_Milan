#!/bin/bash
# Tiêu chí 1.3 – So sánh tốc độ xử lý: máy chủ truyền thống (pandas 1 máy) vs dịch vụ cloud (BigQuery).
# Điều kiện: đã upload 14 file thô vào gs://$BUCKET/raw/milan/
# Thời gian pandas lấy từ meta.json (mục etl.seconds) khi chạy prepare_data.py trên máy.
set -e; source "$(dirname "$0")/config.sh"
bq --location=asia-southeast1 mk -f -d fl5g || true
echo "== Nạp dữ liệu thô vào BigQuery =="
time bq --location=asia-southeast1 load --replace --source_format=CSV --field_delimiter=tab \
  fl5g.raw_milan "gs://$BUCKET/raw/milan/*.txt" \
  cell:INT64,time_ms:INT64,country:INT64,sms_in:FLOAT64,sms_out:FLOAT64,call_in:FLOAT64,call_out:FLOAT64,internet:FLOAT64
echo "== Gộp theo (ô, giờ) bằng SQL – tương đương bước ETL pandas =="
time bq --location=asia-southeast1 query --use_legacy_sql=false --destination_table=fl5g.hourly --replace \
'SELECT cell,
        TIMESTAMP_TRUNC(TIMESTAMP_MILLIS(time_ms), HOUR, "Europe/Rome") AS hour,
        SUM(IFNULL(sms_in,0)+IFNULL(sms_out,0)) AS sms,
        SUM(IFNULL(call_in,0)+IFNULL(call_out,0)) AS call,
        SUM(IFNULL(internet,0)) AS internet
 FROM fl5g.raw_milan GROUP BY cell, hour'
bq show --format=prettyjson fl5g.raw_milan | grep -E '"numRows"|"numBytes"'
echo "Ghi lại 2 thời gian 'real' ở trên + thời gian xử lý (Elapsed) trong BigQuery console → so với etl.seconds của pandas."
