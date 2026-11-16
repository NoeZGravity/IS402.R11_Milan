# Triển khai trên Google Cloud (đồ án Cloud)

Khuyên dùng **Cloud Shell** (nút `>_` trên console.cloud.google.com) – có sẵn gcloud, bq, python.
Upload thư mục `flower_5g` (đã chạy `python prepare_api.py` trên máy) lên Cloud Shell: menu ⋮ → Upload → chọn file zip, rồi `unzip`.

| Bước | Lệnh | Tiêu chí rubric |
|---|---|---|
| 0 | Sửa `cloud/config.sh` (PROJECT_ID, PREFIX) | |
| 1 | `bash cloud/setup_gcp.sh` – bật dịch vụ, 3 bucket (SG/US/EU), lifecycle | 2.3, 3.4 |
| 2 | `bash cloud/deploy_etl.sh` rồi upload file thô: `gcloud storage cp *.txt gs://$BUCKET/raw/milan/` | 3.2 ETL tự động |
| 3 | `python cloud/bench_storage.py --prefix <PREFIX> --raw sms-call-internet-mi-2013-11-01.txt` | 3.1 đọc ghi, 3.3 RTT, 3.4 tối ưu |
| 4 | `bash cloud/bigquery_compare.sh` | 1.3 so sánh máy truyền thống vs cloud |
| 5 | `bash cloud/deploy_api.sh` (3 region) | 4.2 API, 3.3 độ trễ |
| 6 | `bash cloud/deploy_web.sh`, mở web → tab *Cấu hình API* dán 3 URL | 4.4 web gọi API |
| 7 | Web → tab *Độ trễ theo region* → *Đo độ trễ*; tab *Kết quả thí nghiệm* | 3.3, 4.3 |

**Chụp màn hình làm minh chứng:** danh sách bucket, log của hàm ETL, thư mục `processed/`, BigQuery job, 3 service Cloud Run, trang web, Billing → Reports (chi phí), IAM & Admin → Quotas.

**Tiết kiệm chi phí:** Cloud Run chỉ tính tiền khi có request; xoá file `bench/big.bin` sau khi đo; đặt Budget alert (Billing → Budgets) 5–10 USD.
