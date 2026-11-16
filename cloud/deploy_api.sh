#!/bin/bash
# Bước 3 – deploy API (Cloud Run – PaaS) ở 3 region để đo độ trễ theo region.
# Trước đó chạy trên máy:  python prepare_api.py   (chép model + dữ liệu vào api/)
set -e; source "$(dirname "$0")/config.sh"
for R in $REGIONS; do
  gcloud run deploy fl5g-api --source "$(dirname "$0")/../api" --region "$R" \
    --allow-unauthenticated --memory 1Gi --cpu 1 --set-env-vars REGION="$R" --quiet
  echo "URL $R: $(gcloud run services describe fl5g-api --region "$R" --format='value(status.url)')"
done
