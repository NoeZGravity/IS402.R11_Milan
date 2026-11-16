# ==== Sửa các biến này cho đúng project của bạn, rồi chạy:  source cloud/config.sh ====
export PROJECT_ID="wise-invention-514907-t0"          # ID project GCP
export PREFIX="fl5g-vinh-2026"                 # tiền tố tên bucket (phải duy nhất toàn cầu)
export MAIN_REGION="asia-southeast1"      # Singapore – region chính
export REGIONS="asia-southeast1 us-central1 europe-west1"
export BUCKET="${PREFIX}-sg"              # bucket chính (Singapore)
gcloud config set project "$PROJECT_ID"
