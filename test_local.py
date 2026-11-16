"""Kiểm tra nhanh KHÔNG qua Flower: dữ liệu, mô hình, nén/giải nén, 2 vòng FedAvg thủ công.
Nếu file này chạy được mà `flwr run` vẫn lỗi -> lỗi nằm ở phần Flower/Ray, gửi log cho mình.
   python test_local.py
"""
import traceback, torch
print("1) Import Flower apps ...")
try:
    import fl5g.client_app, fl5g.server_app
    print("   OK")
except Exception:
    traceback.print_exc()

from fl5g.task import DATA_DIR, load_clients, make_model, train_local, evaluate_all
from fl5g.compress import encode, decode, state_to_vec, vec_to_state, payload_bytes
print("2) Dữ liệu ở", DATA_DIR)
clients, meta = load_clients()
print(f"   {len(clients)} trạm, train mỗi trạm ~{len(clients[0]['train'][1])} mẫu, nhóm: {sorted({c['group'] for c in clients})}")
cfg = {"hidden": 64, "layers": 2, "cell": "gru", "learning-rate": 1e-3, "batch-size": 64, "local-epochs": 1}
model = make_model(cfg)
print("3) RMSE ban đầu:", round(evaluate_all(model)["rmse"], 4))
for method in ["none", "int8", "int4", "topk:0.01", "topk_int8:0.01"]:
    gsd = make_model(cfg).state_dict(); w = state_to_vec(gsd); n = w.numel()
    for rnd in range(2):
        agg = torch.zeros(n); up = 0
        for c in clients[:4]:
            m = make_model(cfg); m.load_state_dict(vec_to_state(w, gsd))
            train_local(m, c["train"], cfg)
            d = state_to_vec(m.state_dict()) - w
            p, s = encode(d, method); up += payload_bytes(p)
            agg += decode(p, s, method, n) / 4
        w = w + agg
    m = make_model(cfg); m.load_state_dict(vec_to_state(w, gsd))
    print(f"4) {method:15s} RMSE sau 2 vòng = {evaluate_all(m)['rmse']:.4f} | uplink vòng cuối = {up/1e3:.1f} KB")
print("Xong – phần dữ liệu / mô hình / nén hoạt động bình thường.")
