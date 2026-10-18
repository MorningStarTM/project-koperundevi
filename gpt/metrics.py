"""Phase 7 (minimal) -- expose training metrics for Prometheus to scrape.

Only the rank-0 process should call start(); Grafana/Prometheus server setup
itself is a separate, later step -- this just makes the numbers available.
"""

from prometheus_client import Gauge, start_http_server

step = Gauge("train_step", "Current training step")
loss_train = Gauge("train_loss", "Training loss")
loss_val = Gauge("val_loss", "Validation loss")
tokens_per_sec = Gauge("train_tokens_per_sec", "Training throughput in tokens/sec")
step_seconds = Gauge("train_step_seconds", "Wall-clock time per training step")


def start(port=8000):
    start_http_server(port)
    print(f"metrics: http://localhost:{port}/metrics")
