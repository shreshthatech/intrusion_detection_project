"""
Task 70: Lightweight REST API + Dashboard for incidents, tracks, and latency.

Endpoints:
  - GET /            -> HTML dashboard with tiles
  - GET /api/incidents -> JSON list of fused incidents
  - GET /api/health    -> Latest edge health metrics
  - GET /api/drift     -> Latest drift monitor result

This uses the files you already generated:
  - data/processed/api/fused_timeline.jsonl
  - data/processed/edge_metrics/edge_health_log.jsonl
  - data/processed/edge_metrics/drift_log.jsonl
"""

from pathlib import Path
from flask import Flask, jsonify, render_template_string
import json
import os
from typing import List, Dict, Any

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FUSED_TIMELINE = PROJECT_ROOT / "data" / "processed" / "api" / "fused_timeline.jsonl"
EDGE_HEALTH_LOG = PROJECT_ROOT / "data" / "processed" / "edge_metrics" / "edge_health_log.jsonl"
DRIFT_LOG = PROJECT_ROOT / "data" / "processed" / "edge_metrics" / "drift_log.jsonl"


# -------------- Helper functions to load JSONL data -------------- #

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def get_last_entry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    last = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last or {}


# -------------- API endpoints -------------- #

@app.route("/api/incidents")
def api_incidents():
    """
    Return fused incidents from fused_timeline.jsonl
    """
    events = load_jsonl(FUSED_TIMELINE)
    return jsonify({
        "count": len(events),
        "events": events
    })


@app.route("/api/health")
def api_health():
    """
    Return latest edge health metrics.
    """
    last = get_last_entry(EDGE_HEALTH_LOG)
    return jsonify(last)


@app.route("/api/drift")
def api_drift():
    """
    Return latest drift monitor info.
    """
    last = get_last_entry(DRIFT_LOG)
    return jsonify(last)


# -------------- HTML Dashboard -------------- #

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Intrusion IDS Dashboard</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #0f172a;
      color: #e5e7eb;
      margin: 0;
      padding: 20px;
    }
    h1 {
      margin-bottom: 10px;
    }
    .subtitle {
      color: #9ca3af;
      margin-bottom: 20px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }
    .card {
      background: #111827;
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.4);
      border: 1px solid #1f2937;
    }
    .card h2 {
      font-size: 0.95rem;
      margin: 0 0 8px;
      color: #9ca3af;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .card .value {
      font-size: 1.8rem;
      font-weight: bold;
      margin-bottom: 4px;
    }
    .card .label {
      font-size: 0.85rem;
      color: #6b7280;
    }
    .tag {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 0.75rem;
      margin-top: 4px;
    }
    .tag-ok {
      background: #10b98122;
      color: #6ee7b7;
      border: 1px solid #10b98155;
    }
    .tag-warn {
      background: #f59e0b22;
      color: #fbbf24;
      border: 1px solid #f59e0b55;
    }
    .tag-bad {
      background: #ef444422;
      color: #fecaca;
      border: 1px solid #ef444455;
    }
    .footer {
      margin-top: 20px;
      font-size: 0.8rem;
      color: #6b7280;
    }
    a {
      color: #60a5fa;
      text-decoration: none;
    }
  </style>
</head>
<body>
  <h1>Intrusion Detection – Edge Dashboard</h1>
  <div class="subtitle">Multimodal IDS · Vision · Audio · RF · Tracking · Edge Health</div>

  <div class="grid">
    <div class="card">
      <h2>Total Incidents</h2>
      <div class="value">{{ incidents_count }}</div>
      <div class="label">From fused_timeline.jsonl</div>
    </div>

    <div class="card">
      <h2>Edge Performance</h2>
      <div class="value">{{ fps }} FPS</div>
      <div class="label">CPU: {{ cpu }}% · RAM: {{ ram }}%</div>
    </div>

    <div class="card">
      <h2>Drift Status</h2>
      <div class="value">{{ drift_level or "unknown" }}</div>
      {% if drift_level == "none" %}
        <span class="tag tag-ok">Stable</span>
      {% elif drift_level == "moderate" %}
        <span class="tag tag-warn">Monitor</span>
      {% elif drift_level == "major" %}
        <span class="tag tag-bad">Recalibrate</span>
      {% endif %}
      <div class="label">KS={{ ks_stat }} · p={{ ks_p }} · PSI={{ psi }}</div>
    </div>

    <div class="card">
      <h2>Active Model Version</h2>
      <div class="value">{{ active_model }}</div>
      <div class="label">Managed by canary + rollback logic</div>
    </div>
  </div>

  <div class="footer">
    API Endpoints:
    <a href="/api/incidents">/api/incidents</a> ·
    <a href="/api/health">/api/health</a> ·
    <a href="/api/drift">/api/drift</a>
  </div>
</body>
</html>
"""


@app.route("/")
def dashboard():
    # Load incidents
    incidents = load_jsonl(FUSED_TIMELINE)
    incidents_count = len(incidents)

    # Load last health
    health = get_last_entry(EDGE_HEALTH_LOG)
    fps = health.get("fps", 0)
    cpu = health.get("cpu_percent", 0)
    ram = health.get("ram_percent", 0)

    # Load last drift
    drift = get_last_entry(DRIFT_LOG)
    drift_level = drift.get("drift_level", None)
    ks_stat = drift.get("ks_stat", 0.0)
    ks_p = drift.get("ks_p_value", 1.0)
    psi = drift.get("psi", 0.0)

    # Active model – from Task 68 we ended on v2, we can just show that here.
    active_model = "v2"

    return render_template_string(
        TEMPLATE,
        incidents_count=incidents_count,
        fps=fps,
        cpu=cpu,
        ram=ram,
        drift_level=drift_level,
        ks_stat=f"{ks_stat:.3f}" if isinstance(ks_stat, (int, float)) else ks_stat,
        ks_p=f"{ks_p:.3f}" if isinstance(ks_p, (int, float)) else ks_p,
        psi=f"{psi:.3f}" if isinstance(psi, (int, float)) else psi,
        active_model=active_model,
    )


if __name__ == "__main__":
    # Run on http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
