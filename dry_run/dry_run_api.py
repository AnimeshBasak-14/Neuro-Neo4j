"""
dry_run_api.py
==============
FastAPI smoke test using httpx.AsyncClient (no server startup required).

Author: basakanimesh14
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def test_api_startup() -> bool:
    """
    Test all FastAPI endpoints using ASGI transport (no live server).

    Tests
    -----
    - GET  /health    → 200 or 503 (degraded is acceptable in dry run)
    - GET  /metrics   → 200
    - POST /predict   → 200
    - GET  /          → 200 or 307 redirect
    - GET  /metrics/prometheus → 200

    Returns
    -------
    bool  True if all critical tests pass
    """
    t0 = time.time()
    try:
        import httpx
        from api.main import app

        dummy_payload = {
            "subject_id": "sub_dry_run",
            "edge_list": [
                {"source": "Hippocampus_L", "target": "Hippocampus_R", "weight": 0.72},
                {"source": "Frontal_Sup_L", "target": "Frontal_Sup_R", "weight": 0.45},
                {"source": "Thalamus_L",    "target": "Thalamus_R",    "weight": 0.61},
                {"source": "Amygdala_L",    "target": "Amygdala_R",    "weight": 0.38},
            ]
        }

        results: list[tuple[str, bool, str]] = []

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test"
        ) as client:

            # 1. GET /health
            try:
                r = await client.get("/health")
                ok = r.status_code in (200, 503)  # 503 = degraded but reachable
                results.append(("GET /health", ok, f"status={r.status_code}"))
            except Exception as e:
                results.append(("GET /health", False, str(e)))

            # 2. GET /metrics
            try:
                r = await client.get("/metrics")
                ok = r.status_code == 200
                results.append(("GET /metrics", ok, f"status={r.status_code}"))
            except Exception as e:
                results.append(("GET /metrics", False, str(e)))

            # 3. POST /predict
            try:
                r = await client.post("/predict", json=dummy_payload)
                ok = r.status_code == 200
                if ok:
                    body = r.json()
                    ok = "prediction" in body and "label" in body
                results.append(("POST /predict", ok, f"status={r.status_code}"))
            except Exception as e:
                results.append(("POST /predict", False, str(e)))

            # 4. GET / (redirect to /docs)
            try:
                r = await client.get("/", follow_redirects=True)
                ok = r.status_code == 200
                results.append(("GET /", ok, f"status={r.status_code}"))
            except Exception as e:
                results.append(("GET /", False, str(e)))

            # 5. GET /metrics/prometheus
            try:
                r = await client.get("/metrics/prometheus")
                ok = r.status_code == 200 and "neuro_neo4j_macro_f1" in r.text
                results.append(("GET /metrics/prometheus", ok, f"status={r.status_code}"))
            except Exception as e:
                results.append(("GET /metrics/prometheus", False, str(e)))

        elapsed = time.time() - t0
        all_pass = all(ok for _, ok, _ in results)

        for endpoint, ok, detail in results:
            icon = "✅" if ok else "❌"
            print(f"    {icon} {endpoint:<30} {detail}")

        status = "✅ ALL PASS" if all_pass else "⚠️  SOME FAILED"
        print(f"  {status} | {elapsed*1000:.1f}ms")
        return all_pass

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  ❌ API startup FAILED: {exc} | {elapsed*1000:.1f}ms")
        import traceback; traceback.print_exc()
        return False


def run() -> bool:
    """Synchronous entry point for the master orchestrator."""
    return asyncio.run(test_api_startup())
