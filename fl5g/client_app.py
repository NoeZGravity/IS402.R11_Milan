"""ClientApp – chạy tại mỗi Edge node (trạm phát).

Bước 2.2 (On-edge Training): nhận mô hình toàn cục, huấn luyện trên dữ liệu của trạm.
Bước 2.3 (Communication constraints): nén Δw (có error feedback) rồi mới gửi lên Cloud.
"""
from collections import OrderedDict

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl5g.compress import decode, encode, payload_bytes, state_to_vec
from fl5g.task import load_clients, make_model, train_local

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    cfg = context.run_config
    pid = int(context.node_config["partition-id"])
    clients, _ = load_clients()
    data = clients[pid]["train"]

    # 1) Model broadcasting: nhận mô hình toàn cục
    model = make_model(cfg)
    global_sd = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(global_sd)
    w0 = state_to_vec(global_sd)

    # 2) Huấn luyện tại Edge
    mu = float(cfg["proximal-mu"]) if str(cfg["algorithm"]) == "fedprox" else 0.0
    loss = train_local(model, data, cfg, mu)

    # 3) Nén cập nhật Δw (+ error feedback lưu trong trạng thái của node)
    method = str(cfg["compress"])
    delta = state_to_vec(model.state_dict()) - w0
    if method != "none" and bool(cfg["error-feedback"]):
        if "residual" in context.state:
            delta = delta + context.state["residual"].to_torch_state_dict()["r"]
    payload, scalars = encode(delta, method)
    if method != "none" and bool(cfg["error-feedback"]):
        sent = decode(payload, scalars, method, delta.numel())
        context.state["residual"] = ArrayRecord(OrderedDict(r=(delta - sent).contiguous()))

    metrics = {"num-examples": len(data[1]), "train_loss": float(loss),
               "payload_bytes": payload_bytes(payload), **{k: float(v) for k, v in scalars.items()}}
    content = RecordDict({"arrays": ArrayRecord(OrderedDict((k, v.contiguous()) for k, v in payload.items())),
                          "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
