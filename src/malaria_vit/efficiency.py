import time
import torch

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def measure_latency(model, device="cpu", n=50, warmup=10) -> float:
    model.eval().to(device)
    x = torch.randn(1, 3, 224, 224, device=device)
    for _ in range(warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0  # ms per image

def compute_flops(model):
    """Best-effort GFLOPs; returns None if the counter fails (never raises)."""
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        dev = next(model.parameters()).device
        flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224, device=dev))
        flops.unsupported_ops_warnings(False)
        return float(flops.total()) / 1e9
    except Exception:
        return None

def efficiency_report(model, device="cpu") -> dict:
    return {
        "params_millions": count_params(model) / 1e6,
        "latency_ms": measure_latency(model, device=device),
        "gflops": compute_flops(model),
    }
