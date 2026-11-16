# Chạy fl5g trên Docker (Deployment thật thay cho Simulation)

```
                 flwr run . docker   (máy Windows của bạn, cổng 9093)
                          │
                 ┌────────▼────────┐
                 │   superlink     │  Cloud: ServerApp (CompressedFedAvg), lưu kết quả
                 └────────┬────────┘
          ┌───────────────┼───────────────┐   cổng 9092
   ┌──────▼─────┐  ┌──────▼─────┐   ┌──────▼──────┐
   │supernode-0 │  │supernode-1 │ … │supernode-19 │  Edge: mỗi container = 1 trạm phát
   └────────────┘  └────────────┘   └─────────────┘
```

So với simulation: mỗi trạm là một container riêng, giao tiếp với server qua mạng (gRPC),
nên tắt được từng trạm, giới hạn CPU, đo lưu lượng mạng thật.

## Bước 0 – Kiểm tra phiên bản Flower (quan trọng)
```powershell
pip show flwr
```
Nếu khác 1.36.0 thì sửa `FLWR_VERSION` trong file `docker/.env` cho trùng.

## Bước 1 – Khai báo "federation" Docker cho lệnh flwr (1 lần)
Mở file `C:\Users\PC\.flwr\config.toml` (chạy `flwr config list` để xem đường dẫn), thêm vào cuối:
```toml
[superlink.docker]
address = "127.0.0.1:9093"
insecure = true
```

## Bước 2 – Build và bật cluster
Mở PowerShell trong thư mục `flower_5g\docker`:
```powershell
# Thử nhỏ trước: server + 5 trạm (build lần đầu mất vài phút)
docker compose up -d --build superlink supernode-0 supernode-1 supernode-2 supernode-3 supernode-4
docker compose ps
docker logs -f superlink        # cửa sổ riêng, để xem server
```

## Bước 3 – Chạy huấn luyện
Mở PowerShell khác ở thư mục `flower_5g` (có file pyproject.toml):
```powershell
flwr run . docker --stream --run-config "min-nodes=5"
```
- Chạy đủ 20 trạm: `docker compose up -d` (bật hết) rồi `flwr run . docker --stream` (min-nodes = 0 nghĩa là chờ đủ 20).
- Kết quả JSON + mô hình nằm ở `C:\Users\PC\fl5g_results\docker\`.
- Vẽ biểu đồ cho riêng bản Docker:
  `$env:FL5G_OUT="C:\Users\PC\fl5g_results\docker"; python plot_results.py`

## Bước 4 – Các thí nghiệm hệ phân tán
Chạy 6 trạm, cho phép tối thiểu 4: bật superlink + supernode-0..5, rồi
`flwr run . docker --stream --run-config "min-nodes=4 num-server-rounds=30"`

| # | Làm gì (cửa sổ khác, khi đang train) | Quan sát / ghi lại |
|---|---|---|
| 1 | `docker stop supernode-2` | Log server: số client mỗi vòng giảm, có vòng chậm hoặc báo lỗi rồi chạy tiếp. RMSE cuối so với lúc đủ node |
| 2 | `docker start supernode-2` | Node kết nối lại, được chọn ở các vòng sau |
| 3 | `docker update --cpus 0.2 supernode-1` | Thời gian mỗi vòng (`elapsed_s` trong JSON) tăng: FedAvg đồng bộ phải chờ trạm chậm nhất (straggler) |
| 4 | `docker stats` | Cột NET I/O của từng container: lưu lượng mạng thật. So `compress='none'` với `compress='int8'` / `'topk_int8:0.01'` |
| 5 | Dừng hết, chỉ để 3 trạm, đặt `min-nodes=5` | Server đứng chờ: minh hoạ điều kiện số node tối thiểu |

Đổi cấu hình như cũ, chỉ thêm tên federation `docker`:
```powershell
flwr run . docker --stream --run-config "min-nodes=5 compress='int8'"
flwr run . docker --stream --run-config "min-nodes=5 algorithm='fedprox' proximal-mu=0.01"
```

## Tắt
```powershell
docker compose stop      # tắt tạm
docker compose down      # xoá container (image và kết quả vẫn còn)
```

## Lỗi hay gặp
- `flwr run` báo không kết nối được: superlink chưa chạy, hoặc thiếu mục `[superlink.docker]` ở Bước 1.
- Server đứng im "waiting for nodes": số trạm đang bật < `min-nodes` (hoặc < 20 nếu min-nodes = 0).
- Log supernode báo lỗi version: `FLWR_VERSION` trong `.env` khác `pip show flwr` → sửa rồi `docker compose build --no-cache`.
- Hết RAM: mỗi trạm ~300–500MB khi train, 20 trạm cần ~8–10GB → chạy 5–6 trạm là đủ cho thí nghiệm.

Đã sửa nhỏ trong code (bản cũ lưu ở *.bak): thêm tham số `min-nodes` vào pyproject.toml và server_app.py
để chạy được với ít trạm hơn 20. Mặc định 0 = giữ nguyên hành vi cũ.
