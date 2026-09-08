"""
generate_edgelist.py
====================
ETL Generate Stage: Convert synthetic connectomes → unified edge list CSV.

Scans data/synthetic/connectomes/ for per-subject CSV files, applies the
ConnectomeTransformer pipeline to each, and concatenates results into:
  data/processed/edge_list.csv
  data/processed/subjects.csv

Author: basakanimesh14
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neo4j.etl.transform import ConnectomeTransformer
from data.synthetic.generate_synthetic import AAL116_REGIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_edgelist")


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_edge_list(
    connectomes_dir: str | Path,
    subjects_csv: str | Path,
    output_dir: str | Path,
    preprocessing_config: dict | None = None,
) -> pd.DataFrame:
    """
    Process all subject connectomes into a unified edge list.

    Parameters
    ----------
    connectomes_dir      : Directory containing per-subject CSV connectomes
    subjects_csv         : Path to subjects metadata CSV
    output_dir           : Output directory for edge_list.csv
    preprocessing_config : Preprocessing config dict (from config.yaml)

    Returns
    -------
    pd.DataFrame  Full concatenated edge list
    """
    connectomes_dir = Path(connectomes_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load subject metadata
    subjects_df = pd.read_csv(subjects_csv)
    logger.info("Loaded %d subjects from %s", len(subjects_df), subjects_csv)

    # Initialise transformer with AAL-116 regions
    transformer = ConnectomeTransformer(region_names=AAL116_REGIONS)

    all_edge_lists: list[pd.DataFrame] = []
    n_processed = 0
    n_failed = 0

    for _, row in subjects_df.iterrows():
        subject_id = row["PatientID"]
        diagnosis  = row["Diagnosis"]
        matrix_path = connectomes_dir / f"{subject_id}.csv"

        if not matrix_path.exists():
            logger.warning("Connectome not found for subject=%s — skipping", subject_id)
            n_failed += 1
            continue

        try:
            edge_list = transformer.run_pipeline(
                matrix_path=matrix_path,
                subject_id=subject_id,
                diagnosis=diagnosis,
                config=preprocessing_config,
            )
            all_edge_lists.append(edge_list)
            n_processed += 1
        except Exception as exc:
            logger.error("Failed to process subject=%s: %s", subject_id, exc)
            n_failed += 1

    if not all_edge_lists:
        raise RuntimeError("No edge lists generated. Check input data.")

    combined = pd.concat(all_edge_lists, ignore_index=True)
    edge_list_path = output_dir / "edge_list.csv"
    combined.to_csv(edge_list_path, index=False)
    logger.info(
        "Edge list saved → %s  (%d edges, %d subjects processed, %d failed)",
        edge_list_path, len(combined), n_processed, n_failed,
    )

    # Copy subjects metadata
    subjects_out = output_dir / "subjects.csv"
    shutil.copy2(subjects_csv, subjects_out)
    logger.info("Subjects metadata copied → %s", subjects_out)

    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate unified edge list from synthetic connectomes"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)

    generate_edge_list(
        connectomes_dir=PROJECT_ROOT / config["data"]["connectomes_dir"],
        subjects_csv=PROJECT_ROOT / config["data"]["synthetic_dir"] / "subjects.csv",
        output_dir=PROJECT_ROOT / config["data"]["processed_dir"],
        preprocessing_config=config.get("preprocessing", {}),
    )
    logger.info("Edge list generation complete.")


if __name__ == "__main__":
    main()
