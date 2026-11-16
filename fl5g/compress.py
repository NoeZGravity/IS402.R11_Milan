"""Nén / giải nén cập nhật mô hình (Δw) để gửi thật qua Flower.

Client gửi một ArrayRecord chứa các tensor ĐÃ NÉN (int8, int4 đóng gói, top-k ...),
nên số byte truyền đi là số byte thật của payload, không phải ước tính.

method: none | fp16 | int8 | int4 | topk:<p> | topk_int8:<p>   (p: tỉ lệ phần tử giữ lại, ví dụ 0.01)
"""
import math
import torch


def state_to_vec(sd):
    return torch.cat([v.detach().float().flatten() for v in sd.values()])


def vec_to_state(vec, like):
    out, i = {}, 0
    for k, v in like.items():
        n = v.numel()
        out[k] = vec[i:i + n].view_as(v).to(v.dtype).clone()
        i += n
    return out


def payload_bytes(payload):
    return int(sum(t.numel() * t.element_size() for t in payload.values()))


def _pack_int4(q):
    """q trong [-7, 7] -> 2 giá trị / 1 byte (uint8)."""
    u = (q + 8).to(torch.uint8)
    if u.numel() % 2:
        u = torch.cat([u, torch.zeros(1, dtype=torch.uint8)])
    return u[0::2] | (u[1::2] << 4)


def _unpack_int4(p, n):
    lo = (p & 0x0F).to(torch.int16) - 8
    hi = ((p >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.stack([lo, hi], 1).flatten()[:n]
    return q.float()


def encode(vec, method):
    """vec (float32, 1 chiều) -> (payload: dict[str, Tensor], scalars: dict[str, float])."""
    n = vec.numel()
    if method == "none":
        return {"d": vec.float()}, {}
    if method == "fp16":
        return {"d": vec.half()}, {}
    if method in ("int8", "int4"):
        b = int(method[3:]); qmax = 2 ** (b - 1) - 1
        s = float(vec.abs().max().clamp_min(1e-12) / qmax)
        q = torch.round(vec / s).clamp(-qmax, qmax)
        if b == 8:
            return {"q": q.to(torch.int8)}, {"scale": s}
        return {"q": _pack_int4(q.to(torch.int16))}, {"scale": s}
    if method.startswith("topk"):
        kind, p = method.split(":")
        k = max(1, int(n * float(p)))
        idx = vec.abs().topk(k).indices
        vals = vec[idx]
        if kind == "topk_int8":
            s = float(vals.abs().max().clamp_min(1e-12) / 127)
            return {"idx": idx.to(torch.int32), "q": torch.round(vals / s).clamp(-127, 127).to(torch.int8)}, {"scale": s}
        return {"idx": idx.to(torch.int32), "v": vals.float()}, {}
    raise ValueError(method)


def decode(payload, scalars, method, n):
    """Ngược lại encode -> vector float32 độ dài n."""
    if method in ("none", "fp16"):
        return payload["d"].float()
    if method == "int8":
        return payload["q"].float() * scalars["scale"]
    if method == "int4":
        return _unpack_int4(payload["q"], n) * scalars["scale"]
    if method.startswith("topk"):
        out = torch.zeros(n)
        idx = payload["idx"].long()
        if method.startswith("topk_int8"):
            out[idx] = payload["q"].float() * scalars["scale"]
        else:
            out[idx] = payload["v"].float()
        return out
    raise ValueError(method)
