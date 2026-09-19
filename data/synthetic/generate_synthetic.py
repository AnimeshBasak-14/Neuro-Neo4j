"""
generate_synthetic.py
=====================
Generates synthetic fMRI functional connectivity data using the AAL-116 atlas.

Atlas   : Automated Anatomical Labeling (AAL) — 116 regions
Subjects: N healthy + N diseased (configurable via CLI)
Outputs :
  - data/synthetic/subjects.csv          (metadata: PatientID, Diagnosis, Age, ScannerSite)
  - data/synthetic/connectomes/<id>.csv  (116×116 Pearson correlation matrix)

Diseased connectivity perturbations (simulating neurodegeneration):
  - Reduced inter-hemispheric long-range correlations
  - Increased local (same-lobe) correlations
  - Added scanner-site SNR noise

Author: basakanimesh14
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_synthetic")

# ---------------------------------------------------------------------------
# AAL-116 Region Names (official labels)
# ---------------------------------------------------------------------------
AAL116_REGIONS: list[str] = [
    "Precentral_L", "Precentral_R",
    "Frontal_Sup_L", "Frontal_Sup_R",
    "Frontal_Sup_Orb_L", "Frontal_Sup_Orb_R",
    "Frontal_Mid_L", "Frontal_Mid_R",
    "Frontal_Mid_Orb_L", "Frontal_Mid_Orb_R",
    "Frontal_Inf_Oper_L", "Frontal_Inf_Oper_R",
    "Frontal_Inf_Tri_L", "Frontal_Inf_Tri_R",
    "Frontal_Inf_Orb_L", "Frontal_Inf_Orb_R",
    "Rolandic_Oper_L", "Rolandic_Oper_R",
    "Supp_Motor_Area_L", "Supp_Motor_Area_R",
    "Olfactory_L", "Olfactory_R",
    "Frontal_Sup_Medial_L", "Frontal_Sup_Medial_R",
    "Frontal_Med_Orb_L", "Frontal_Med_Orb_R",
    "Rectus_L", "Rectus_R",
    "Insula_L", "Insula_R",
    "Cingulum_Ant_L", "Cingulum_Ant_R",
    "Cingulum_Mid_L", "Cingulum_Mid_R",
    "Cingulum_Post_L", "Cingulum_Post_R",
    "Hippocampus_L", "Hippocampus_R",
    "ParaHippocampal_L", "ParaHippocampal_R",
    "Amygdala_L", "Amygdala_R",
    "Calcarine_L", "Calcarine_R",
    "Cuneus_L", "Cuneus_R",
    "Lingual_L", "Lingual_R",
    "Occipital_Sup_L", "Occipital_Sup_R",
    "Occipital_Mid_L", "Occipital_Mid_R",
    "Occipital_Inf_L", "Occipital_Inf_R",
    "Fusiform_L", "Fusiform_R",
    "Postcentral_L", "Postcentral_R",
    "Parietal_Sup_L", "Parietal_Sup_R",
    "Parietal_Inf_L", "Parietal_Inf_R",
    "SupraMarginal_L", "SupraMarginal_R",
    "Angular_L", "Angular_R",
    "Precuneus_L", "Precuneus_R",
    "Paracentral_Lobule_L", "Paracentral_Lobule_R",
    "Caudate_L", "Caudate_R",
    "Putamen_L", "Putamen_R",
    "Pallidum_L", "Pallidum_R",
    "Thalamus_L", "Thalamus_R",
    "Heschl_L", "Heschl_R",
    "Temporal_Sup_L", "Temporal_Sup_R",
    "Temporal_Pole_Sup_L", "Temporal_Pole_Sup_R",
    "Temporal_Mid_L", "Temporal_Mid_R",
    "Temporal_Pole_Mid_L", "Temporal_Pole_Mid_R",
    "Temporal_Inf_L", "Temporal_Inf_R",
    "Cerebelum_Crus1_L", "Cerebelum_Crus1_R",
    "Cerebelum_Crus2_L", "Cerebelum_Crus2_R",
    "Cerebelum_3_L", "Cerebelum_3_R",
    "Cerebelum_4_5_L", "Cerebelum_4_5_R",
    "Cerebelum_6_L", "Cerebelum_6_R",
    "Cerebelum_7b_L", "Cerebelum_7b_R",
    "Cerebelum_8_L", "Cerebelum_8_R",
    "Cerebelum_9_L", "Cerebelum_9_R",
    "Cerebelum_10_L", "Cerebelum_10_R",
    "Vermis_1_2", "Vermis_3",
    "Vermis_4_5", "Vermis_6",
    "Vermis_7", "Vermis_8",
    "Vermis_9", "Vermis_10",
]

assert len(AAL116_REGIONS) == 116, f"Expected 116 regions, got {len(AAL116_REGIONS)}"

# AAL-116 lobe assignments (approximate, for disease perturbation)
# Indices are 0-based
LOBE_GROUPS: dict[str, list[int]] = {
    "frontal":    list(range(0, 28)),
    "limbic":     list(range(28, 42)),
    "occipital":  list(range(42, 56)),
    "parietal":   list(range(56, 70)),
    "subcortical":list(range(70, 80)),
    "temporal":   list(range(80, 90)),
    "cerebellar": list(range(90, 116)),
}

SCANNER_SITES: list[str] = ["site_A", "site_B", "site_C"]
SITE_NOISE_STD: dict[str, float] = {"site_A": 0.02, "site_B": 0.04, "site_C": 0.06}


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------

def _make_random_spd_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random symmetric positive-definite correlation matrix."""
    A = rng.standard_normal((n, n))
    cov = A @ A.T + n * np.eye(n)
    # Normalise to correlation
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    return corr


def _apply_disease_perturbation(
    corr: np.ndarray,
    inter_factor: float = 0.4,
    local_boost: float = 1.3,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Simulate neurodegenerative connectivity breakdown.

    Strategy
    --------
    1. Reduce inter-hemispheric (L↔R) correlations by *inter_factor*.
    2. Boost within-lobe (local) correlations by *local_boost*.
    3. Add small Gaussian noise.
    """
    perturbed = corr.copy()
    n = corr.shape[0]

    # Reduce inter-hemispheric: L regions are even indices, R are odd
    left_idx  = list(range(0, 90, 2))   # even-index regions (0,2,4,...88)
    right_idx = list(range(1, 90, 2))   # odd-index regions
    for li, ri in zip(left_idx, right_idx):
        perturbed[li, ri] *= inter_factor
        perturbed[ri, li] *= inter_factor

    # Boost within-lobe connectivity
    for lobe, indices in LOBE_GROUPS.items():
        for i in indices:
            for j in indices:
                if i != j:
                    perturbed[i, j] = min(1.0, perturbed[i, j] * local_boost)

    # Re-symmetrise and clip
    perturbed = (perturbed + perturbed.T) / 2
    np.fill_diagonal(perturbed, 1.0)
    perturbed = np.clip(perturbed, -1.0, 1.0)
    return perturbed


def _add_site_noise(corr: np.ndarray, site: str, rng: np.random.Generator) -> np.ndarray:
    """Add scanner-site specific Gaussian noise to simulate SNR variance."""
    std = SITE_NOISE_STD.get(site, 0.03)
    noise = rng.normal(0, std, corr.shape)
    noise = (noise + noise.T) / 2   # keep symmetric
    np.fill_diagonal(noise, 0.0)
    noisy = np.clip(corr + noise, -1.0, 1.0)
    return noisy


def generate_subject_connectome(
    subject_id: str,
    diagnosis: str,
    site: str,
    rng: np.random.Generator,
    n_regions: int = 116,
) -> np.ndarray:
    """
    Generate a single subject's 116×116 functional connectivity matrix.

    Parameters
    ----------
    subject_id : str  (unused directly, for logging)
    diagnosis  : str  'Healthy' or 'Diseased'
    site       : str  Scanner site identifier
    rng        : np.random.Generator
    n_regions  : int  Number of brain regions (default 116 for AAL-116)

    Returns
    -------
    np.ndarray  (n_regions × n_regions) symmetric correlation matrix
    """
    corr = _make_random_spd_matrix(n_regions, rng)

    if diagnosis == "Diseased":
        corr = _apply_disease_perturbation(corr, rng=rng)

    corr = _add_site_noise(corr, site, rng)
    return corr


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_dataset(
    n_healthy: int = 25,
    n_diseased: int = 25,
    n_regions: int = 116,
    random_seed: int = 42,
    output_dir: str = "data/synthetic",
) -> pd.DataFrame:
    """
    Generate synthetic fMRI connectivity data for a full cohort.

    Parameters
    ----------
    n_healthy  : Number of healthy control subjects
    n_diseased : Number of diseased subjects
    n_regions  : Number of brain regions (116 for AAL-116)
    random_seed: NumPy random seed for reproducibility
    output_dir : Root output directory

    Returns
    -------
    pd.DataFrame : Subject metadata DataFrame
    """
    rng = np.random.default_rng(random_seed)

    out_path = Path(output_dir)
    conn_path = out_path / "connectomes"
    conn_path.mkdir(parents=True, exist_ok=True)

    region_names = AAL116_REGIONS[:n_regions]
    subjects: list[dict] = []
    n_subjects = n_healthy + n_diseased

    logger.info(
        "Generating %d subjects (%d healthy, %d diseased) | "
        "Atlas: AAL-%d | Seed: %d",
        n_subjects, n_healthy, n_diseased, n_regions, random_seed,
    )

    for idx in range(n_subjects):
        subject_id  = f"sub_{idx + 1:04d}"
        diagnosis   = "Healthy" if idx < n_healthy else "Diseased"
        age         = int(rng.integers(45, 81))
        site        = rng.choice(SCANNER_SITES)

        corr = generate_subject_connectome(subject_id, diagnosis, site, rng, n_regions)

        # Save correlation matrix
        corr_df = pd.DataFrame(corr, index=region_names, columns=region_names)
        corr_df.to_csv(conn_path / f"{subject_id}.csv", float_format="%.6f")

        subjects.append({
            "PatientID":   subject_id,
            "Diagnosis":   diagnosis,
            "Label":       1 if diagnosis == "Diseased" else 0,
            "Age":         age,
            "ScannerSite": site,
        })

        if (idx + 1) % 10 == 0:
            logger.info("  Generated %d / %d subjects ...", idx + 1, n_subjects)

    metadata_df = pd.DataFrame(subjects)
    metadata_path = out_path / "subjects.csv"
    metadata_df.to_csv(metadata_path, index=False)

    logger.info("Metadata saved  → %s", metadata_path)
    logger.info("Connectomes saved → %s", conn_path)
    logger.info(
        "Class distribution: %s",
        metadata_df["Diagnosis"].value_counts().to_dict(),
    )
    return metadata_df


# ---------------------------------------------------------------------------
# Convenience wrapper for train_launcher.py
# ---------------------------------------------------------------------------

def make_connectome(
    diagnosis:    str,
    scanner_site: str,
    seed:         int,
    n_regions:    int = 116,
) -> np.ndarray:
    """
    Seed-based wrapper around generate_subject_connectome.

    Parameters
    ----------
    diagnosis    : 'Healthy' or 'Diseased'
    scanner_site : e.g. 'site_A'
    seed         : Integer seed for reproducibility
    n_regions    : 116 for AAL-116

    Returns
    -------
    np.ndarray  (n_regions × n_regions) symmetric correlation matrix
    """
    rng = np.random.default_rng(seed)
    return generate_subject_connectome(
        subject_id=f"sub_{seed:04d}",
        diagnosis=diagnosis,
        site=scanner_site,
        rng=rng,
        n_regions=n_regions,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic AAL-116 fMRI connectivity data for Neuro-Neo4j"
    )
    parser.add_argument("--n-healthy",  type=int, default=25, help="Number of healthy subjects")
    parser.add_argument("--n-diseased", type=int, default=25, help="Number of diseased subjects")
    parser.add_argument("--n-regions",  type=int, default=116, help="Number of brain regions (≤116)")
    parser.add_argument("--seed",       type=int, default=42,  help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        n_healthy=args.n_healthy,
        n_diseased=args.n_diseased,
        n_regions=args.n_regions,
        random_seed=args.seed,
        output_dir=args.output_dir,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
