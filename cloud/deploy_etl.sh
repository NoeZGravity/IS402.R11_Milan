#!/bin/bash
# Bước 2 – deploy hàm ETL (FaaS). Sau đó upload file thô vào gs://$BUCKET/raw/milan/ là tự động xử lý.
set -e; source "$(dirname "$0")/config.sh"
gcloud functions deploy fl5g-etl --gen2 --runtime=python312 --region="$MAIN_REGION" \
  --source="$(dirname "$0")/etl" --entry-point=etl --memory=4GiB --timeout=540s \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=$BUCKET"
echo "Thử: gcloud storage cp sms-call-internet-mi-2013-11-01.txt gs://$BUCKET/raw/milan/"
echo "Xem kết quả: gcloud storage ls gs://$BUCKET/processed/milan/hourly/"
