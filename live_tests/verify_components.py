"""Machine-checked evidence matrix over the 16 system components (README "system components").

Reads a scenario's history.jsonl (written by live_tests.harness.History) and requires at least
one 'ok' (or, for a deliberately-expected-failure check, a 'matched': true) event whose step name
starts with each component's declared prefixes. A component the scenario never touches is reported
as NOT_REACHED, not silently passed -- per /connect-pipeline, "not reachable" is itself a finding.

Usage: python -m live_tests.verify_components <scenario_out_dir> [<scenario_out_dir> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

COMPONENTS = {
    1: ("Auth + RBAC", ["rbac."]),
    2: ("Idempotency keys", ["idempotency."]),
    3: ("Dataset registry", ["dataset."]),
    4: ("Model registry lifecycle", ["model."]),
    5: ("Policy publish/decide/gate", ["policy."]),
    6: ("Audit chain export+verify+tamper", ["audit."]),
    7: ("Drift check (real windows)", ["drift."]),
    8: ("Incidents + escalation", ["incident."]),
    9: ("Lineage", ["lineage."]),
    10: ("MLflow mirror", ["mlflow."]),
    11: ("SBOM / supply chain", ["sbom."]),
    12: ("Config/manifest lint", ["lint."]),
    13: ("Metrics/telemetry", ["telemetry."]),
    14: ("CLI + SDK", ["sdk.", "cli."]),
    15: ("Calibration cross-check", ["calibration."]),
    16: ("Durability (kill -9 + restart)", ["durability."]),
}


def load_events(out_dir: Path) -> list[dict]:
    p = out_dir / "history.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _touched(events: list[dict], prefixes: list[str]) -> list[dict]:
    return [e for e in events if any(e["step"].startswith(p) for p in prefixes)]


def evidence_ok(e: dict) -> bool:
    if e["outcome"] == "ok":
        return True
    if e["outcome"] == "info":
        d = e.get("detail", {})
        res = d.get("result")
        if isinstance(res, dict) and res.get("matched") is True:
            return True
    return False


def verify(out_dir: Path) -> dict:
    events = load_events(out_dir)
    rows = {}
    for num, (name, prefixes) in COMPONENTS.items():
        touched = _touched(events, prefixes)
        ok_events = [e for e in touched if evidence_ok(e)]
        if not touched:
            rows[num] = {"name": name, "status": "NOT_REACHED", "events": 0}
        elif ok_events:
            rows[num] = {"name": name, "status": "EVIDENCED", "events": len(touched), "ok": len(ok_events)}
        else:
            rows[num] = {"name": name, "status": "TOUCHED_BUT_FAILED", "events": len(touched),
                        "sample": touched[0]}
    evidenced = sum(1 for r in rows.values() if r["status"] == "EVIDENCED")
    crashes = [e for e in events if e["outcome"] == "crash"]
    return {"scenario_dir": str(out_dir), "total_events": len(events), "crashes": len(crashes),
            "components_evidenced": f"{evidenced}/{len(COMPONENTS)}", "rows": rows,
            "crash_steps": [e["step"] for e in crashes]}


def main():
    for d in sys.argv[1:]:
        result = verify(Path(d))
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
