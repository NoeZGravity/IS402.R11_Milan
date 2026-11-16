#!/bin/bash
# Bước 4 – đưa trang web lên Cloud Storage dạng website tĩnh (công khai).
set -e; source "$(dirname "$0")/config.sh"
WEB="${PREFIX}-web"
gcloud storage buckets create "gs://$WEB" --location="$MAIN_REGION" --uniform-bucket-level-access || true
gcloud storage buckets add-iam-policy-binding "gs://$WEB" --member=allUsers --role=roles/storage.objectViewer
gcloud storage buckets update "gs://$WEB" --web-main-page-suffix=index.html
gcloud storage cp "$(dirname "$0")/../web/index.html" "gs://$WEB/index.html" --cache-control="no-cache"
echo "Mở web: https://storage.googleapis.com/$WEB/index.html  (vào tab 'Cấu hình API' dán 3 URL Cloud Run)"
