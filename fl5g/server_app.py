"""ServerApp – chạy trên Cloud.

Chiến lược CompressedFedAvg: nhận payload đã nén từ các Edge node, giải nén Δw,
lấy trung bình có trọng số theo số mẫu, cập nhật mô hình toàn cục.
Ghi lại theo từng vòng: RMSE, số byte nhận (uplink thật), số byte phát (downlink).
"""
import json
import time
from collections import OrderedDict

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl5g.compress import decode, state_to_vec, vec_to_state
from fl5g.task import OUT_DIR, evaluate_all, load_clients, make_model

app = ServerApp()


class CompressedFedAvg(FedAvg):
    def __init__(self, method, **kw):
        super().__init__(**kw)
        self.method = method
        self.global_sd = None
        self.log = []          # thông tin theo vòng
        self._t = time.time()

    def configure_train(self, server_round, arrays, config, grid):
        self.global_sd = arrays.to_torch_state_dict()
        msgs = list(super().configure_train(server_round, arrays, config, grid))
        n_params = sum(v.numel() for v in self.global_sd.values())
        self._down = len(msgs) * n_params * 4          # phát mô hình fp32 xuống từng node
        return msgs

    def aggregate_train(self, server_round, replies):
        replies = [r for r in replies if not r.has_error()]
        if not replies:
            return None, None
        w = state_to_vec(self.global_sd); n = w.numel()
        tot = sum(r.content["metrics"]["num-examples"] for r in replies)
        agg = torch.zeros(n); up = 0; losses = []
        for r in replies:
            m = r.content["metrics"]
            payload = r.content["arrays"].to_torch_state_dict()
            scalars = {"scale": m["scale"]} if "scale" in m else {}
            agg += (m["num-examples"] / tot) * decode(payload, scalars, self.method, n)
            up += int(m["payload_bytes"]); losses.append(m["train_loss"])
        new_sd = vec_to_state(w + agg, self.global_sd)
        self.log.append({"round": server_round, "clients": len(replies), "uplink_bytes": up,
                         "downlink_bytes": self._down, "train_loss": sum(losses) / len(losses),
                         "elapsed_s": time.time() - self._t})
        return ArrayRecord(OrderedDict(new_sd)), MetricRecord({"uplink_bytes": up, "train_loss": sum(losses) / len(losses)})


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    clients, meta = load_clients()
    model = make_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    # Chỉ in ký tự ASCII: console Windows (cp1252) không in được tiếng Việt có dấu
    print(f"Model {cfg['cell'].upper()}: {n_params} params ({n_params*4/1024:.1f} KB fp32) | "
          f"{len(clients)} edge nodes | compress = {cfg['compress']} | algorithm = {cfg['algorithm']}")

    evals = []

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        m = make_model(cfg); m.load_state_dict(arrays.to_torch_state_dict())
        res = evaluate_all(m, "test"); res["round"] = server_round
        evals.append(res)
        return MetricRecord({k: float(v) for k, v in res.items()})

    strategy = CompressedFedAvg(
        method=str(cfg["compress"]),
        fraction_train=float(cfg["fraction-train"]),
        fraction_evaluate=0.0,
        min_train_nodes=2,
        # min-nodes = 0 -> chờ đủ tất cả các trạm (như chạy simulation)
        min_available_nodes=int(cfg.get("min-nodes", 0)) or len(clients),
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(model.state_dict()),
        train_config=ConfigRecord({}),
        num_rounds=int(cfg["num-server-rounds"]),
        evaluate_fn=global_evaluate,
    )

    # Lưu kết quả
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg['algorithm']}_{str(cfg['compress']).replace(':', '-')}"
    out = {"tag": tag, "config": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in cfg.items()},
           "n_params": n_params, "rounds": strategy.log, "eval": evals,
           "total_uplink_MB": sum(r["uplink_bytes"] for r in strategy.log) / 1e6,
           "total_downlink_MB": sum(r["downlink_bytes"] for r in strategy.log) / 1e6,
           "final_rmse": evals[-1]["rmse"] if evals else None}
    json.dump(out, open(OUT_DIR / f"{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    torch.save(result.arrays.to_torch_state_dict(), OUT_DIR / f"{tag}_model.pt")
    print(f"\nSaved {OUT_DIR / (tag + '.json')}  | final RMSE = {out['final_rmse']:.4f} | "
          f"uplink = {out['total_uplink_MB']:.2f} MB | downlink = {out['total_downlink_MB']:.2f} MB")
