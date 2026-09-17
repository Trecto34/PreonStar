#!/usr/bin/env python3
"""Attach the independent compatibility result to an inner-loop report."""
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
gate_status = int(sys.argv[2])
evidence_name = sys.argv[3]

if report_path.exists():
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        report = {}
else:
    report = {}

report.setdefault("hypothesis", "Candidate optimization was evaluated by the local Karpathy worker.")
report.setdefault("target", "Swift Qwen3.8-27B IQ3_XXS on AMD BC-250 Vulkan")
evidence = report.setdefault("evidence", [])
if evidence_name not in evidence:
    evidence.append(evidence_name)
gates = report.setdefault("gates", {})
gates["compatibility_swift_iq3_xxs"] = gate_status == 0
gates["compatibility_qwen35_qwen36_moe"] = gate_status == 0

if gate_status != 0:
    report["decision"] = "rejected"
    report["reason"] = "Independent Swift/Qwen3.5-Qwen3.6 compatibility gate failed."
elif report.get("decision") not in ("accepted", "rejected"):
    report["decision"] = "rejected"
    report["reason"] = "Worker did not produce an accepted or rejected decision."

report_path.write_text(json.dumps(report, indent=2) + "\n")
