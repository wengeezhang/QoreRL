"""Small, dependency-free viewer for the learner's opt-in recipe probes."""

from __future__ import annotations

import base64
import ipaddress
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

_MAX_METRICS_BYTES = 16 * 1024 * 1024
_MAX_CONFIG_BYTES = 1024 * 1024
_VARIANTS = ("bf16", "mxfp8", "mxfp8_hadamard")


def _number(value: object, *, integer: bool = False) -> bool:
    return (
        isinstance(value, int if integer else (int, float))
        and not isinstance(value, bool)
        and (isinstance(value, int) or math.isfinite(value))
        and value >= 0
    )


def _record(value: object) -> dict:
    """Keep the public probe fields and reject data the charts cannot interpret."""
    if (
        not isinstance(value, dict)
        or type(value.get("schema")) is not int
        or value.get("schema") != 1
        or value.get("kind") != "recipe_probe"
    ):
        raise ValueError("expected a schema 1 recipe_probe object")
    if value.get("source") not in ("measured", "demo"):
        raise ValueError("source must be measured or demo")
    if value.get("status") not in ("ok", "error", "skipped"):
        raise ValueError("status must be ok, error, or skipped")
    result = {key: value[key] for key in ("schema", "kind", "source", "status")}
    for key in ("step", "weight_version", "optimizer_steps", "layer", "num_layers"):
        if not _number(value.get(key), integer=True):
            raise ValueError(f"{key} must be a nonnegative integer")
        if value[key] > 2**53 - 1:
            raise ValueError(f"{key} exceeds the viewer's integer range")
        result[key] = value[key]
    if not 0 < result["num_layers"] <= 4096 or result["layer"] >= result["num_layers"]:
        raise ValueError("layer must be within num_layers (1–4096)")
    for key in ("recipe_id", "module", "phase"):
        if not isinstance(value.get(key), str) or len(value[key]) > 512:
            raise ValueError(f"{key} must be a string of at most 512 characters")
        result[key] = value[key]
    shape = value.get("shape_mnk")
    if shape is not None:
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or not all(_number(n, integer=True) and n <= 2**53 - 1 for n in shape)
            or shape[1] == 0
            or shape[2] == 0
            or (shape[0] == 0 and value["status"] == "ok")
        ):
            raise ValueError("shape_mnk requires positive M,N,K; M may be zero for non-ok events")
        result["shape_mnk"] = shape
    for key in ("original_rows", "repeats", "probe_seconds"):
        number = value.get(key)
        if number is not None:
            if not _number(number, integer=key != "probe_seconds") or number > 2**53 - 1:
                raise ValueError(f"{key} must be a finite nonnegative number")
            result[key] = number
    variants = value.get("variants", {})
    if not isinstance(variants, dict):
        raise ValueError("variants must be an object")
    result["variants"] = {}
    for name in _VARIANTS:
        variant = variants.get(name)
        if variant is None:
            continue
        if not isinstance(variant, dict):
            raise ValueError(f"{name} must be an object")
        result["variants"][name] = {}
        for key in ("latency_ms", "output_relative_l2"):
            number = variant.get(key)
            if number is not None:
                if not _number(number) or number > 2**53 - 1:
                    raise ValueError(f"{name}.{key} must be a finite nonnegative number")
                result["variants"][name][key] = number
    if "error" in value:
        result["error"] = str(value["error"])[:2000]
    return result


def load_report(run_dir: Path) -> dict:
    """Read only config.json and recipe_metrics.jsonl, tolerating an active writer.

    The final unterminated JSONL line is left for the next poll. Completed invalid
    lines are reported. Very large histories are bounded to their final 16 MiB;
    the visible warning explains that earlier layer observations may be absent.
    """
    run_dir = Path(run_dir)
    report: dict = {"records": [], "metadata": {}, "warnings": []}
    warnings = report["warnings"]
    config = run_dir / "config.json"
    try:
        with config.open("rb") as stream:
            raw_config = stream.read(_MAX_CONFIG_BYTES + 1)
        if len(raw_config) > _MAX_CONFIG_BYTES:
            warnings.append("config.json exceeds 1 MiB; metadata was not loaded.")
        else:
            metadata = json.loads(raw_config)
            if not isinstance(metadata, dict):
                raise ValueError("expected an object")
            json.dumps(metadata, allow_nan=False)
            report["metadata"] = metadata
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        warnings.append(f"Could not read config.json: {type(exc).__name__}.")
    try:
        with (run_dir / "recipe_metrics.jsonl").open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            offset = max(0, size - _MAX_METRICS_BYTES)
            stream.seek(offset)
            if offset:
                stream.readline()  # Discard the partial first line of the retained tail.
                warnings.append(
                    "History exceeds 16 MiB; showing the recent tail. Earlier layers may be absent."
                )
            raw = stream.read(_MAX_METRICS_BYTES)
    except FileNotFoundError:
        return report
    except OSError as exc:
        warnings.append(f"Could not read recipe_metrics.jsonl: {type(exc).__name__}.")
        return report
    lines = raw.split(b"\n")[:-1]
    invalid_count = 0
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            report["records"].append(_record(value))
        except (UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
            invalid_count += 1
            if invalid_count <= 20:
                location = f"tail line {index}" if offset else f"line {index}"
                reason = (
                    str(exc) if not isinstance(exc, (json.JSONDecodeError, UnicodeError)) else "invalid JSON"
                )
                warnings.append(f"recipe_metrics.jsonl {location}: {reason}.")
    if invalid_count > 20:
        warnings.append(f"{invalid_count - 20} additional completed lines were invalid.")
    return report


def _html(report: dict, *, live: bool) -> str:
    template = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
    # HTML's script parser processes closing tags even in application/json scripts.
    payload = json.dumps(report, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    mark = Path(__file__).with_name("assets").joinpath("qorerl-mark.png").read_bytes()
    logo = "data:image/png;base64," + base64.b64encode(mark).decode("ascii")
    return (
        template.replace("__QORERL_LIVE__", str(live).lower(), 1)
        .replace("__QORERL_REPORT__", payload, 1)
        .replace("__QORERL_LOGO__", logo, 1)
    )


def export_report(run_dir: Path, output: Path) -> Path:
    """Write a standalone report with embedded data and no network dependencies."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_html(load_report(run_dir), live=False), encoding="utf-8")
    return output


def _make_server(run_dir: Path, host: str, port: int) -> ThreadingHTTPServer:
    if host != "localhost":
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if not loopback or ":" in host:
            raise ValueError("The recipe viewer binds to localhost or an IPv4 loopback address only.")
    run_dir = Path(run_dir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                body = _html(load_report(run_dir), live=True).encode("utf-8")
                content_type = "text/html; charset=utf-8"
            elif path == "/api/report":
                body = json.dumps(load_report(run_dir), ensure_ascii=True, allow_nan=False).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(404, "Not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "img-src data: blob:; connect-src 'self'; font-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(run_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the dashboard and one JSON endpoint until interrupted."""
    server = _make_server(Path(run_dir), host, port)
    print(f"QoreRL Recipe Insights: http://{host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
