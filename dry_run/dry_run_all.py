"""
dry_run_all.py
==============
Neuro-Neo4j Master Dry Run Orchestrator.

Runs all smoke tests sequentially and prints a rich summary table.
No Neo4j instance, GPU, or trained model required.

Usage
-----
    uv run python dry_run/dry_run_all.py

Author: basakanimesh14
"""

from __future__ import annotations

import sys
import time
import os
from pathlib import Path

# Force UTF-8 output on Windows to handle all characters safely
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Rich console (fallback to plain print if not installed)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    _HAS_RICH = True
    # Force UTF-8 on rich console to avoid Windows CP1252 charmap issues
    console = Console(highlight=False, markup=True)
except ImportError:
    _HAS_RICH = False
    console = None

PASS_MARK = "[PASS]"
FAIL_MARK = "[FAIL]"


def _print_header() -> None:
    if _HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]Neuro-Neo4j[/bold cyan] -- Master Dry Run\n"
            "[dim]Author: basakanimesh14[/dim]",
            border_style="cyan",
        ))
    else:
        print("=" * 60)
        print("  Neuro-Neo4j -- Master Dry Run")
        print("  Author: basakanimesh14")
        print("=" * 60)


def _print_summary(results: list[tuple[str, bool, float, str]]) -> int:
    """Print summary table. Returns exit code (0=all pass, 1=any fail)."""
    n_pass = sum(1 for _, ok, _, _ in results if ok)
    n_fail = len(results) - n_pass

    if _HAS_RICH:
        table = Table(
            title="Dry Run Results",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
        )
        table.add_column("#",       style="dim",    width=4)
        table.add_column("Test",    style="white",  width=35)
        table.add_column("Status",  width=8)
        table.add_column("Time",    width=10, justify="right")
        table.add_column("Error",   style="red dim", width=28)

        for i, (name, ok, dur, err) in enumerate(results, 1):
            status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
            table.add_row(
                str(i), name, status,
                f"{dur*1000:.0f}ms",
                err[:28] if err else "",
            )

        console.print(table)
        if n_fail == 0:
            console.print(f"\n[bold green]All {n_pass} tests PASSED[/bold green]")
        else:
            console.print(f"\n[bold red]{n_fail} test(s) FAILED -- see errors above[/bold red]")
            console.print("[dim]Tip: missing libraries are the most common cause.[/dim]")
    else:
        print("\n" + "=" * 72)
        print(f"{'#':<4} {'Test':<35} {'Status':<8} {'Time':>8}  {'Error'}")
        print("-" * 72)
        for i, (name, ok, dur, err) in enumerate(results, 1):
            status = "PASS" if ok else "FAIL"
            print(f"{i:<4} {name:<35} {status:<8} {dur*1000:>6.0f}ms  {err[:20] if err else ''}")
        print("=" * 72)
        if n_fail == 0:
            print(f"All {n_pass} tests PASSED")
        else:
            print(f"{n_fail} test(s) FAILED")

    return 0 if n_fail == 0 else 1


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _run_test(name: str, fn) -> tuple[str, bool, float, str]:
    """Run a single test function and capture result."""
    t0 = time.time()
    try:
        ok = fn()
        elapsed = time.time() - t0
        return (name, bool(ok), elapsed, "")
    except Exception as exc:
        elapsed = time.time() - t0
        return (name, False, elapsed, str(exc)[:60])


def main() -> None:
    _print_header()
    print()

    results: list[tuple[str, bool, float, str]] = []

    # ---- 1. ETL Pipeline ----
    print("[1/10] ETL Pipeline...")
    from dry_run.dry_run_etl import test_etl_pipeline
    results.append(_run_test("ETL Pipeline", test_etl_pipeline))

    # ---- 2. GDS Mock ----
    print("[2/10] GDS Feature Engineering (mock)...")
    from dry_run.dry_run_gds import test_gds_mock
    results.append(_run_test("GDS Mock", test_gds_mock))

    # ---- 3. SVM Baseline ----
    print("[3/10] SVM Baseline...")
    from dry_run.dry_run_baselines import test_svm_dry_run
    results.append(_run_test("SVM Baseline", test_svm_dry_run))

    # ---- 4. RF-GDV Baseline ----
    print("[4/10] RF-GDV Baseline...")
    from dry_run.dry_run_baselines import test_rf_gdv_dry_run
    results.append(_run_test("RF-GDV Baseline", test_rf_gdv_dry_run))

    # ---- 5. node2vec Baseline ----
    print("[5/10] node2vec Baseline...")
    from dry_run.dry_run_baselines import test_node2vec_dry_run
    results.append(_run_test("node2vec Baseline", test_node2vec_dry_run))

    # ---- 6. GNN Forward Pass ----
    print("[6/10] GNN Forward Pass...")
    from dry_run.dry_run_gnn import test_model_forward_pass
    results.append(_run_test("GNN Forward Pass", test_model_forward_pass))

    # ---- 7. Dataset Loading ----
    print("[7/10] Dataset Loading...")
    from dry_run.dry_run_gnn import test_dataset_loading
    results.append(_run_test("Dataset Loading", test_dataset_loading))

    # ---- 8. Trainer Dry Run ----
    print("[8/10] GNN Trainer Dry Run...")
    from dry_run.dry_run_gnn import test_trainer_dry_run
    results.append(_run_test("GNN Trainer Dry Run", test_trainer_dry_run))

    # ---- 9. GNNExplainer ----
    print("[9/10] GNNExplainer...")
    from dry_run.dry_run_explainer import test_explainer_dry_run
    results.append(_run_test("GNNExplainer", test_explainer_dry_run))

    # ---- 10. FastAPI ----
    print("[10/10] FastAPI Endpoints...")
    from dry_run.dry_run_api import run as test_api
    results.append(_run_test("FastAPI Endpoints", test_api))

    print()
    exit_code = _print_summary(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
