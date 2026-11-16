# Một image dùng chung cho SuperLink (Cloud) và SuperNode (Edge).
# Phiên bản flwr PHẢI trùng với flwr trên máy bạn: xem bằng  pip show flwr
FROM python:3.12-slim
ARG FLWR_VERSION=1.36.0
# torch bản CPU (~200MB thay vì vài GB bản CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "flwr==${FLWR_VERSION}" numpy pandas pyarrow scikit-learn matplotlib
WORKDIR /app
ENV FL5G_DATA=/data \
    FL5G_OUT=/results \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8
