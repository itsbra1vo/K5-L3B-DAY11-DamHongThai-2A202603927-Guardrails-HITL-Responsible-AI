from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))


def test_rate_limit_is_sliding_and_per_user(monkeypatch):
    from assignment import rate_limiter

    now = [100.0]
    monkeypatch.setattr(rate_limiter.time, "time", lambda: now[0])
    limiter = rate_limiter.RateLimitPlugin(max_requests=2, window_seconds=10)

    async def send(user_id):
        return await limiter.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id=user_id),
            user_message=None,
        )

    assert asyncio.run(send("alice")) is None
    now[0] = 101.0
    assert asyncio.run(send("alice")) is None
    now[0] = 102.0
    assert asyncio.run(send("alice")) is not None
    assert asyncio.run(send("bob")) is None
    now[0] = 111.0
    assert asyncio.run(send("alice")) is None
    assert limiter.blocked_count == 1


def test_audit_and_monitoring_export_json(tmp_path):
    from assignment.audit_log import AuditLogPlugin
    from assignment.monitoring import MonitoringAlert

    audit = AuditLogPlugin()
    audit.record_input(user_id="user-1", text="What is my balance?", request_id="r1")
    audit.record_output(
        user_id="user-1",
        text="Request blocked",
        blocked=True,
        layer="input_guardrail",
        request_id="r1",
    )
    audit_path = tmp_path / "nested" / "audit.json"
    audit.export_json(str(audit_path))
    audit_entries = json.loads(audit_path.read_text(encoding="utf-8"))
    assert len(audit_entries) == 2
    assert audit_entries[1]["latency_ms"] >= 0
    assert audit_entries[1]["layer"] == "input_guardrail"

    monitor = MonitoringAlert()
    monitor.total_requests = 4
    monitor.blocked_requests = 3
    monitor.rate_limit_hits = 6
    monitor.judge_checks = 2
    monitor.judge_fails = 1
    assert {alert.metric for alert in monitor.check_metrics()} == {
        "block_rate",
        "rate_limit_hits",
        "judge_fail_rate",
    }
    metrics_path = tmp_path / "metrics.json"
    monitor.export_json(str(metrics_path))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["block_rate"] == 0.75
    assert len(metrics["alerts"]) == 3


def test_pipeline_order_and_strict_egress_rules():
    from assignment.pipeline import build_production_plugins, is_egress_allowed

    plugins = build_production_plugins()
    assert [plugin.name for plugin in plugins] == [
        "rate_limiter",
        "input_guardrail",
        "output_guardrail",
    ]
    assert is_egress_allowed(
        "https://api.vinbank.example/v1/transfers",
        "approved transfer amount 500000",
    )
    for destination, payload in (
        ("https://api.vinbank.example/v1/transfers", "password is secret123"),
        ("https://api.vinbank.example/v1/transfers", "contact user@example.com"),
        ("https://api.vinbank.example/v1/transfers", "call 0901234567"),
        ("https://api.vinbank.example/v1/transfers", "db.vinbank.internal"),
        ("https://api.vinbank.example.evil/v1/transfers", "approved transfer"),
        ("https://user@api.vinbank.example/v1/transfers", "approved transfer"),
    ):
        assert not is_egress_allowed(destination, payload)
