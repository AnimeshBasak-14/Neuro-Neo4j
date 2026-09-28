"""
render_math_assets.py
======================
Renders mathematical equations into crisp, high-resolution, transparent LaTeX PNG equations
using Matplotlib's MathText engine with pure Computer Modern mathematical syntax.

Author: Animesh Basak (IIT Roorkee)
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt

MATH_DIR = Path("assets/math")
MATH_DIR.mkdir(parents=True, exist_ok=True)

NAVY  = "#0e346e"
DARK  = "#1e293b"
RED   = "#b91c1c"
GREEN = "#107848"

EQUATIONS = {
    # Part 1: Problem Statement
    "eq1_pearson": (
        r"$r_{ij} = \frac{\sum_{t=1}^T (x_i(t) - \bar{x}_i)(x_j(t) - \bar{x}_j)}"
        r"{\sqrt{\sum_{t=1}^T (x_i(t) - \bar{x}_i)^2} \sqrt{\sum_{t=1}^T (x_j(t) - \bar{x}_j)^2}} \in [-1, 1]$",
        18, NAVY, 7.0, 0.95
    ),
    "eq1_sparsification": (
        r"$A_{ij} = |r_{ij}| \;\; \mathrm{if} \; |r_{ij}| \geq \tau \;\; (\tau = 0.3), \quad A_{ij} = 0 \;\; \mathrm{otherwise}, \quad A_{ii} = 0$",
        17, NAVY, 7.2, 0.85
    ),
    "eq2_classification": (
        r"$P(y_k = 1 \mid G_k) = \sigma( f_{\theta}(A_k, X_k) ) = \frac{1}{1 + e^{-f_{\theta}(A_k, X_k)}}, \quad y_k \in \{0, 1\}$",
        17, NAVY, 6.8, 0.85
    ),

    # Part 2: Baselines
    "base_svm": (
        r"$\max_{\alpha} \sum_{k=1}^N \alpha_k - \frac{1}{2} \sum_{k,l} \alpha_k \alpha_l y_k y_l \exp(-\gamma \|\mathbf{v}_k - \mathbf{v}_l\|_2^2), \quad 0 \leq \alpha_k \leq C$",
        17, RED, 7.8, 0.85
    ),
    "base_rf_gdv": (
        r"$\mathrm{GDV}(u) = [u_0, u_1, \dots, u_d]^T, \quad \Phi(G) = [\mathrm{mean}(\mathrm{GDV}), \mathrm{std}(\mathrm{GDV}), \mathrm{max}(\mathrm{GDV})] \in \mathbb{R}^{15}$",
        17, RED, 7.8, 0.85
    ),
    "base_node2vec": (
        r"$P(c_i = x \mid c_{i-1} = v) = \frac{\pi_{vx}}{Z}, \quad \alpha_{pq}(t, x) = \frac{1}{p} \; (d_{tx}=0), \;\; 1 \; (d_{tx}=1), \;\; \frac{1}{q} \; (d_{tx}=2)$",
        17, RED, 7.5, 0.9
    ),
    "eq4_gatv1": (
        r"$\alpha_{ij}^{\mathrm{GATv1}} = \frac{\exp(\mathrm{LeakyReLU}(\mathbf{a}^T [\mathbf{W} h_i \,\|\, \mathbf{W} h_j]))}"
        r"{\sum_{k \in \mathcal{N}_i} \exp(\mathrm{LeakyReLU}(\mathbf{a}^T [\mathbf{W} h_i \,\|\, \mathbf{W} h_k]))}$",
        17, RED, 6.8, 0.95
    ),
    "base_gatv1_defect": (
        r"$\mathbf{a}^T [\mathbf{W} h_i \,\|\, \mathbf{W} h_j] = \mathbf{a}_l^T \mathbf{W} h_i + \mathbf{a}_r^T \mathbf{W} h_j \rightarrow \mathrm{Static \; Attention \; Rank \; Collapse}$",
        16, RED, 7.5, 0.8
    ),

    # Part 3: What We Will Do (Proposed Methodology)
    "eq3_edge_gcn": (
        r"$h_i^{(l+1)} = \mathrm{SiLU}\left( \mathbf{W}_{\mathrm{GCN}}^{(l)} \sum_{j \in \mathcal{N}_i \cup \{i\}} \frac{e_{ij}}{\sqrt{\tilde{d}_i \tilde{d}_j}} \, h_j^{(l)} \right) + h_i^{(l)}$",
        17, NAVY, 6.8, 0.9
    ),
    "eq5_gatv2": (
        r"$\alpha_{ij} = \frac{\exp( \mathbf{a}^T \mathrm{LeakyReLU}(\mathbf{W}_l h_i + \mathbf{W}_r h_j + \mathbf{W}_e e_{ij}) )}"
        r"{\sum_{k \in \mathcal{N}_i} \exp( \mathbf{a}^T \mathrm{LeakyReLU}(\mathbf{W}_l h_i + \mathbf{W}_r h_k + \mathbf{W}_e e_{ik}) )}$",
        17, NAVY, 6.8, 0.95
    ),
    "eq5_gatv2_update": (
        r"$h_i^{(l+1)} = \mathrm{SiLU}\left( \|_{k=1}^H \sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}_v^k h_j^{(l)} \right) + h_i^{(l)}$",
        17, NAVY, 6.2, 0.85
    ),
    "eq6_dual_pool": (
        r"$h_G = [h_{\mathrm{mean}} \,\|\, h_{\mathrm{max}}] = \left[ \frac{1}{|V|} \sum_{v \in V} h_v \;\vert\vert\; \max_{v \in V}(h_v) \right] \in \mathbb{R}^{2d}$",
        17, NAVY, 7.2, 0.85
    ),
    "eq7_swiglu": (
        r"$\mathrm{SwiGLU}(h_G) = \left( \mathrm{Swish}(h_G \mathbf{W}_{\mathrm{gate}} + \mathbf{b}_{\mathrm{gate}}) \odot (h_G \mathbf{W}_{\mathrm{up}} + \mathbf{b}_{\mathrm{up}}) \right) \mathbf{W}_{\mathrm{down}} + \mathbf{b}_{\mathrm{down}}$",
        17, NAVY, 8.2, 0.85
    ),
    "eq8_youden": (
        r"$J(t) = \mathrm{Sensitivity}(t) + \mathrm{Specificity}(t) - 1 = \mathrm{TPR}(t) - \mathrm{FPR}(t)$",
        17, GREEN, 6.8, 0.8
    ),
    "eq8_youden_opt": (
        r"$t^* = \arg\max_{t \in [0, 1]} J(t), \qquad \hat{y} = \mathbb{I}\left( P(y=1 \mid G) \geq t^* \right)$",
        17, GREEN, 6.8, 0.8
    ),
    "eq9_gnnexplainer": (
        r"$\min_{\mathbf{M}, \mathbf{F}} \mathcal{L}\left( \hat{y}, f(A \odot \sigma(\mathbf{M}), X \odot \sigma(\mathbf{F})) \right) + \lambda_1 \|\sigma(\mathbf{M})\|_1 + \lambda_2 \mathrm{Ent}(\mathbf{M})$",
        17, NAVY, 8.0, 0.85
    ),
    "eq9_entropy": (
        r"$\mathrm{Ent}(\mathbf{M}) = -\sum_{e} \left( \sigma(M_e) \log \sigma(M_e) + (1 - \sigma(M_e)) \log (1 - \sigma(M_e)) \right)$",
        16, NAVY, 7.5, 0.85
    ),
    "ext_multimodal": (
        r"$h_i^{\mathrm{fused}} = \gamma_i \cdot h_i^{\mathrm{func}} + (1 - \gamma_i) \cdot h_i^{\mathrm{struct}}, \quad \gamma_i = \sigma\left( \mathbf{W}_\gamma [h_i^{\mathrm{func}} \,\|\, h_i^{\mathrm{struct}}] + b_\gamma \right)$",
        17, NAVY, 8.2, 0.85
    ),
}

def render_all():
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",  # Computer Modern Standard LaTeX Font
    })

    print(f"Rendering {len(EQUATIONS)} LaTeX mathematical expressions to {MATH_DIR}...")
    for key, (latex_code, fsize, color, width, height) in EQUATIONS.items():
        fig = plt.figure(figsize=(width, height), dpi=300)
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.patch.set_alpha(0.0)
        ax.axis("off")

        ax.text(
            0.5, 0.5,
            latex_code,
            fontsize=fsize,
            color=color,
            ha="center",
            va="center",
            transform=ax.transAxes
        )

        out_path = MATH_DIR / f"{key}.png"
        fig.savefig(out_path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        print(f"  [RENDERED] {out_path.name}")

    print("All LaTeX mathematical equations successfully rendered!")

if __name__ == "__main__":
    render_all()
