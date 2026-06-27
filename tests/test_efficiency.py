from malaria_vit.models import create_model
from malaria_vit.efficiency import count_params, measure_latency, compute_flops, efficiency_report

def test_param_count_positive():
    m = create_model("deit_small_patch16_224", pretrained=False)
    assert count_params(m) > 1_000_000

def test_latency_positive():
    m = create_model("deit_small_patch16_224", pretrained=False)
    assert measure_latency(m, device="cpu", n=2, warmup=1) > 0

def test_flops_float_or_none():
    m = create_model("deit_small_patch16_224", pretrained=False)
    result = compute_flops(m)
    assert result is None or result > 0

def test_efficiency_report_keys():
    m = create_model("deit_small_patch16_224", pretrained=False)
    report = efficiency_report(m, device="cpu")
    assert set(report.keys()) == {"params_millions", "latency_ms", "gflops"}
    assert report["params_millions"] > 0
    assert report["latency_ms"] > 0
