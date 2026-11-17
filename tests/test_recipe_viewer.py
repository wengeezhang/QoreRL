import json
import shutil
import subprocess
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from qorerl.observability import viewer


def probe(**overrides):
    record = {
        "schema": 1,
        "kind": "recipe_probe",
        "source": "measured",
        "status": "ok",
        "step": 8,
        "weight_version": 3,
        "optimizer_steps": 3,
        "recipe_id": "mxfp8_hadamard",
        "layer": 0,
        "num_layers": 2,
        "module": f"model.layers.{overrides.get('layer', 0)}.mlp.down_proj",
        "phase": "learner.fprop",
        "shape_mnk": [128, 256, 512],
        "original_rows": 1024,
        "repeats": 5,
        "variants": {
            "bf16": {"latency_ms": 0.1, "output_relative_l2": 0.0},
            "mxfp8": {"latency_ms": 0.05, "output_relative_l2": 0.04},
            "mxfp8_hadamard": {"latency_ms": 0.08, "output_relative_l2": 0.02},
        },
        "probe_seconds": 0.4,
    }
    return record | overrides


def write_records(directory, *records):
    (directory / "recipe_metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_missing_probe_file_is_an_honest_empty_report(tmp_path):
    (tmp_path / "config.json").write_text('{"model": "test"}')
    report = viewer.load_report(tmp_path)
    assert report == {"records": [], "metadata": {"model": "test"}, "warnings": []}


def test_active_writer_partial_line_is_deferred_but_completed_errors_are_visible(tmp_path):
    metrics = tmp_path / "recipe_metrics.jsonl"
    write_records(tmp_path, probe())
    with metrics.open("ab") as stream:
        stream.write(b'{"broken":}\n')
        stream.write(json.dumps(probe(layer=1)).encode()[:70])
    report = viewer.load_report(tmp_path)
    assert report["records"] == [probe()]
    assert len(report["warnings"]) == 1
    assert "line 2: invalid JSON" in report["warnings"][0]
    # An intact but unterminated final record is also deferred until committed by newline.
    write_records(tmp_path, probe())
    with metrics.open("a") as stream:
        stream.write(json.dumps(probe(layer=1)))
    assert len(viewer.load_report(tmp_path)["records"]) == 1
    with metrics.open("a") as stream:
        stream.write("\n")
    assert len(viewer.load_report(tmp_path)["records"]) == 2


def test_missing_numbers_and_failed_probes_never_become_zero_measurements(tmp_path):
    failed = probe(status="skipped", layer=1, shape_mnk=[0, 256, 512], variants={}, error="No capture")
    incomplete = probe(variants={"mxfp8": {"latency_ms": 0.1}})
    write_records(tmp_path, failed, incomplete)
    report = viewer.load_report(tmp_path)
    assert not report["warnings"]
    assert report["records"][0]["error"] == "No capture"
    assert report["records"][0]["variants"] == {}
    assert report["records"][1]["variants"] == {"mxfp8": {"latency_ms": 0.1}}


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript pairing check requires Node.js")
def test_dashboard_pairs_canonical_producer_paths_without_reusing_stale_probes():
    # Exercise the dashboard's actual data selection, independently of browser layout.
    html = Path(viewer.__file__).with_name("dashboard.html").read_text()
    finite = html[html.index("const finite =") : html.index("const format =")]
    derive = html[html.index("function derive() {") : html.index("function drawChart(")]
    records = [
        probe(num_layers=3),
        probe(layer=1, num_layers=3),
        probe(layer=1, num_layers=3, step=9, status="error", variants={}, error="Probe failed"),
        probe(layer=1, num_layers=3, source="demo"),
        probe(layer=2, num_layers=3, module="model.layers.2.self_attn.q_proj"),
    ]
    script = (
        f"const report = {json.dumps({'records': records})};\n"
        "const state = {source:'measured', baseline:'mxfp8', candidate:'mxfp8_hadamard'};\n"
        + finite
        + derive
        + "console.log(JSON.stringify(derive()));"
    )
    result = subprocess.run(
        [shutil.which("node"), "-"], input=script, capture_output=True, text=True, check=True
    )
    model = json.loads(result.stdout)
    assert len(model["pairs"]) == 1
    assert model["pairs"][0]["module"] == "model.layers.0.mlp.down_proj"
    assert model["total"] == 3
    assert model["rows"][1]["valid"] is False
    assert model["rows"][1]["record"]["step"] == 9
    assert "record" not in model["rows"][2]


@pytest.mark.parametrize(
    "invalid",
    [
        {"variants": {"bf16": {"latency_ms": float("nan")}}},
        {"variants": {"mxfp8": {"output_relative_l2": float("inf")}}},
        {"variants": {"mxfp8": {"latency_ms": True}}},
        {"step": -1},
        {"weight_version": 2**54},
        {"layer": 5},
        {"num_layers": 100_000_000},
        {"shape_mnk": [0, 256, 512]},
        {"shape_mnk": [128, 0, 512]},
        {"source": "unknown"},
        {"schema": True},
    ],
)
def test_invalid_completed_record_is_reported_without_crashing(tmp_path, invalid):
    write_records(tmp_path, probe(**invalid), probe())
    report = viewer.load_report(tmp_path)
    assert report["records"] == [probe()]
    assert len(report["warnings"]) == 1
    assert "line 1" in report["warnings"][0]


def test_recent_history_is_bounded_and_truncation_is_visible(tmp_path, monkeypatch):
    write_records(tmp_path, probe(step=1), probe(step=2), probe(step=3))
    monkeypatch.setattr(viewer, "_MAX_METRICS_BYTES", len(json.dumps(probe())) * 2)
    report = viewer.load_report(tmp_path)
    assert report["records"][-1]["step"] == 3
    assert len(report["records"]) < 3
    assert "recent tail" in report["warnings"][0]


@pytest.mark.parametrize("config", ['{"value": NaN}', '{"value": 1e999}', "[]", "{broken"])
def test_invalid_metadata_does_not_break_report_export(tmp_path, config):
    (tmp_path / "config.json").write_text(config)
    write_records(tmp_path, probe())
    path = viewer.export_report(tmp_path, tmp_path / "report.html")
    assert path.is_file()
    assert viewer.load_report(tmp_path)["metadata"] == {}
    assert viewer.load_report(tmp_path)["warnings"]


def test_standalone_export_escapes_embedded_html_and_has_no_remote_assets(tmp_path):
    payload = '</script><img src=x onerror="alert(1)">'
    write_records(tmp_path, probe(recipe_id=payload))
    (tmp_path / "config.json").write_text(json.dumps({"name": payload}))
    output = viewer.export_report(tmp_path, tmp_path / "nested" / "report.html")
    text = output.read_text()
    assert payload not in text
    assert "\\u003c/script>" in text
    assert 'data-live="false"' in text
    assert "__QORERL_REPORT__" not in text
    assert "__QORERL_LIVE__" not in text
    assert '<script src="http' not in text
    assert '<link href="http' not in text
    assert 'id="export-svg"' in text
    assert 'id="export-png"' in text
    assert "not whole-training speed" in text


def test_server_only_serves_fixed_dashboard_and_data_routes(tmp_path):
    write_records(tmp_path, probe())
    (tmp_path / "private.txt").write_text("must not be served")
    server = viewer._make_server(tmp_path, "127.0.0.1", 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/") as response:
            assert 'data-live="true"' in response.read().decode()
            assert response.headers["Cache-Control"] == "no-store"
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        with urlopen(base + "/api/report?poll=1") as response:
            assert json.load(response)["records"] == [probe()]
        for path in ("/private.txt", "/../private.txt", "/%2e%2e/config.json", "/config.json"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + path)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.0.1", "example.com"])
def test_server_rejects_non_loopback_binding(tmp_path, host):
    with pytest.raises(ValueError, match="loopback"):
        viewer._make_server(tmp_path, host, 0)
