#!/bin/bash
# Bước 1 – bật dịch vụ, tạo bucket ở 3 region, lifecycle rule.  Chạy trong Cloud Shell:  bash cloud/setup_gcp.sh
set -e; source "$(dirname "$0")/config.sh"
gcloud services enable storage.googleapis.com run.googleapis.com cloudfunctions.googleapis.com \
  cloudbuild.googleapis.com artifactregistry.googleapis.com eventarc.googleapis.com pubsub.googleapis.com bigquery.googleapis.com
declare -A SUF=([asia-southeast1]=sg [us-central1]=us [europe-west1]=eu)
for R in $REGIONS; do
  B="${PREFIX}-${SUF[$R]}"
  gcloud storage buckets create "gs://$B" --location="$R" --uniform-bucket-level-access || echo "(bucket $B đã có)"
done
# Tối ưu chi phí: kết quả cũ > 30 ngày chuyển sang lớp NEARLINE, dữ liệu thô > 60 ngày sang COLDLINE
cat > /tmp/lc.json <<'J'
{"rule":[
 {"action":{"type":"SetStorageClass","storageClass":"NEARLINE"},"condition":{"age":30,"matchesPrefix":["results/"]}},
 {"action":{"type":"SetStorageClass","storageClass":"COLDLINE"},"condition":{"age":60,"matchesPrefix":["raw/"]}}]}
J
gcloud storage buckets update "gs://$BUCKET" --lifecycle-file=/tmp/lc.json
# Quyền cho Cloud Storage gửi sự kiện tới Eventarc (cần cho ETL tự động)
PN=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:service-$PN@gs-project-accounts.iam.gserviceaccount.com" --role=roles/pubsub.publisher
echo "Xong. Bucket chính: gs://$BUCKET"
