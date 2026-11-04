"""
make_figures.py — exportable figures for Presentation 2 (Neuro-Neo4j).

Every number drawn here is recomputed from files in the repo:
  data/processed/subjects.csv, data/synthetic/connectomes/*.npy,
  data/processed/cached_dataset.pt, checkpoints/best_model_v100.pt,
  logs/biomarker_explanations.json

Run from the repo root (CPU only, no Neo4j needed, ~1-2 min):
    CUDA_VISIBLE_DEVICES= uv run python presentation2/make_figures.py

Outputs 300-DPI PNGs (+ SVG) into presentation2/figures/.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.synthetic.generate_synthetic import AAL116_REGIONS, LOBE_GROUPS  # noqa: E402
from neo4j_layer.etl.transform import ConnectomeTransformer  # noqa: E402
from validation.subject_split import subject_level_split  # noqa: E402

OUT = ROOT / "presentation2" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette: validated default instance (fixed slot order, never cycled) ----
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HEALTHY, DISEASED = "#2a78d6", "#eb6834"          # slot 1, slot 2
LOBE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6"]   # slots 1..7
LOBES = list(LOBE_GROUPS.keys())
LOBE_OF = np.zeros(116, dtype=int)
for k, lobe in enumerate(LOBES):
    LOBE_OF[LOBE_GROUPS[lobe]] = k

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 2,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("saved", OUT / f"{name}.png")


# ---------------------------------------------------------------------------
# Shared data: same preprocessing as scripts/train_launcher.py::step_etl
# ---------------------------------------------------------------------------
subj = pd.read_csv(ROOT / "data/processed/subjects.csv")
T = ConnectomeTransformer(AAL116_REGIONS)
IU = np.triu_indices(116, 1)
mats = []
for pid in subj.PatientID:
    f = next((ROOT / "data/synthetic/connectomes").glob(f"{pid}_*.npy"))
    m = T.threshold(T.normalize(T.regress_global_signal(np.load(f))), 0.3)
    mats.append(m)
mats = np.stack(mats)                       # (500, 116, 116)
X = mats[:, IU[0], IU[1]]                   # (500, 6670) flattened upper triangle
y = subj.Label.values
tr, va, te = subject_level_split(subjects_df=subj, ratios=(0.7, 0.15, 0.15), random_seed=42)
pid2i = {p: i for i, p in enumerate(subj.PatientID)}
itr = np.array([pid2i[p] for p in tr.PatientID])
iva = np.array([pid2i[p] for p in va.PatientID])
ite = np.array([pid2i[p] for p in te.PatientID])
assert not (set(itr) & set(ite)) and not (set(iva) & set(ite))

INTRA = LOBE_OF[IU[0]] == LOBE_OF[IU[1]]
HOMO = (IU[0] < 90) & (IU[0] % 2 == 0) & (IU[1] == IU[0] + 1)


# ---------------------------------------------------------------------------
# Fig 1 — EDA: edge-weight distribution and edge counts per class
# ---------------------------------------------------------------------------
def fig_eda():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(-5, 5, 81)
    for lab, col, name in [(0, HEALTHY, "Healthy"), (1, DISEASED, "Diseased")]:
        w = X[y == lab]
        w = w[w != 0]
        ax[0].hist(w, bins=bins, histtype="step", color=col, linewidth=2, density=True, label=name)
    ax[0].axvspan(-0.3, 0.3, color=GRID, alpha=0.8, lw=0)
    ax[0].text(0, ax[0].get_ylim()[1] * 0.92, "pruned\n|z| < 0.3", ha="center", color=INK2, fontsize=9)
    ax[0].set(title="Retained edge weights (z-scored)", xlabel="edge weight z_ij", ylabel="density")
    ax[0].legend()
    counts = (X != 0).sum(1)
    data = [counts[y == 0], counts[y == 1]]
    bp = ax[1].boxplot(data, widths=0.4, patch_artist=True, medianprops=dict(color=INK))
    for patch, col in zip(bp["boxes"], [HEALTHY, DISEASED]):
        patch.set_facecolor(col); patch.set_alpha(0.35); patch.set_edgecolor(col)
    ax[1].set_xticks([1, 2], ["Healthy", "Diseased"])
    ax[1].set(title="Edges kept per subject (of 6,670)", ylabel="undirected edges")
    save(fig, "fig1_eda_edges")


# ---------------------------------------------------------------------------
# Fig 2 — EDA: where the class signal lives (intra-lobe vs inter-lobe |z|)
# ---------------------------------------------------------------------------
def fig_signal():
    feats = {
        "intra-lobe": np.abs(X[:, INTRA]).mean(1),
        "inter-lobe": np.abs(X[:, ~INTRA]).mean(1),
        "homotopic L-R": np.abs(X[:, HOMO]).mean(1),
    }
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, (name, f) in zip(ax, feats.items()):
        bp = a.boxplot([f[y == 0], f[y == 1]], widths=0.4, patch_artist=True,
                       medianprops=dict(color=INK))
        for patch, col in zip(bp["boxes"], [HEALTHY, DISEASED]):
            patch.set_facecolor(col); patch.set_alpha(0.35); patch.set_edgecolor(col)
        auc = roc_auc_score(y[ite], f[ite])
        auc = max(auc, 1 - auc)
        a.set_xticks([1, 2], ["Healthy", "Diseased"])
        a.set(title=f"mean |z|, {name} edges", ylabel="mean |z_ij|",
              xlabel=f"one-scalar test AUC = {auc:.3f}")
    save(fig, "fig2_eda_signal_location")


# ---------------------------------------------------------------------------
# Fig 3 — group-difference connectome (Diseased minus Healthy), lobe ordered
# ---------------------------------------------------------------------------
def fig_diff_matrix():
    d = np.abs(mats[y == 1]).mean(0) - np.abs(mats[y == 0]).mean(0)
    np.fill_diagonal(d, 0)
    lim = np.percentile(np.abs(d), 99)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.grid(False)
    im = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim)   # diverging, neutral midpoint
    edges = np.cumsum([len(LOBE_GROUPS[l]) for l in LOBES])[:-1] - 0.5
    for e in edges:
        ax.axhline(e, color=INK2, lw=0.5); ax.axvline(e, color=INK2, lw=0.5)
    mids = [np.mean(LOBE_GROUPS[l]) for l in LOBES]
    ax.set_xticks(mids, LOBES, rotation=45, ha="right"); ax.set_yticks(mids, LOBES)
    ax.set_title("Mean |z| difference: Diseased − Healthy")
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("Δ mean |z_ij|")
    save(fig, "fig3_group_difference_matrix")


# ---------------------------------------------------------------------------
# Fig 4 — one subject's brain graph, circular layout grouped by lobe
# ---------------------------------------------------------------------------
def fig_brain_graph(pid="sub_0001", top=250):
    m = mats[pid2i[pid]]
    G = nx.Graph()
    G.add_nodes_from(range(116))
    w = np.abs(m[IU])
    order = np.argsort(-w)[:top]
    for k in order:
        G.add_edge(IU[0][k], IU[1][k], weight=w[k])
    # circular layout in lobe order
    theta = np.linspace(0, 2 * np.pi, 116, endpoint=False)
    pos = {n: (np.cos(theta[n]), np.sin(theta[n])) for n in range(116)}
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis("off"); ax.set_aspect("equal")
    ws = np.array([G[u][v]["weight"] for u, v in G.edges()])
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.3 + 1.2 * (ws - ws.min()) / (np.ptp(ws) + 1e-9),
                           edge_color=INK2, alpha=0.35)
    deg = np.array([G.degree(n) for n in range(116)])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=20 + 12 * deg,
                           node_color=[LOBE_COLORS[LOBE_OF[n]] for n in range(116)],
                           edgecolors=SURFACE, linewidths=1.5)
    for k, lobe in enumerate(LOBES):
        ax.scatter([], [], color=LOBE_COLORS[k], label=lobe, s=60)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    diag = subj.Diagnosis[pid2i[pid]]
    ax.set_title(f"{pid} ({diag}): top {top} edges by |z|, node size = degree")
    save(fig, "fig4_brain_graph_subject")


# ---------------------------------------------------------------------------
# GNN scores from the saved V100 checkpoint
# ---------------------------------------------------------------------------
def gnn_scores():
    import yaml
    from torch_geometric.loader import DataLoader
    from models.gnn.gcn_gat_model import BrainGNNModel
    from models.metrics import find_optimal_threshold
    cfg = yaml.safe_load(open(ROOT / "config/v100_config.yaml"))["model"]
    data = torch.load(ROOT / "data/processed/cached_dataset.pt", weights_only=False)
    by_id = {d.subject_id: d for d in data}
    model = BrainGNNModel(cfg["in_channels"], cfg["hidden_channels"], 2, cfg["num_heads"],
                          cfg["dropout"], cfg["num_gcn_layers"], cfg["num_gat_layers"],
                          True, True, True, True)
    ck = torch.load(ROOT / "checkpoints/best_model_v100.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    def run(pids):
        ys, ps = [], []
        with torch.no_grad():
            for b in DataLoader([by_id[p] for p in pids], batch_size=32):
                ps += torch.softmax(model(b.x, b.edge_index, b.edge_attr, b.batch), -1)[:, 1].tolist()
                ys += b.y.view(-1).tolist()
        return np.array(ys), np.array(ps)

    yv, pv = run(va.PatientID); yt, pt = run(te.PatientID)
    t_val = find_optimal_threshold(yv, pv)[0]
    t_test = find_optimal_threshold(yt, pt)[0]
    return yv, pv, yt, pt, t_val, t_test, ck["history"]


def fig_training(history):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], color=HEALTHY, label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], color=DISEASED, label="validation")
    ax[0].set(title="Cross-entropy loss (label smoothing 0.1)", xlabel="epoch", ylim=(0.55, 0.80))
    ax[0].legend()
    ax[1].plot(ep, [h["train_f1"] for h in history], color=HEALTHY, label="train Macro-F1")
    ax[1].plot(ep, [h["val_f1"] for h in history], color=DISEASED, label="val Macro-F1")
    ax[1].plot(ep, [h["val_auc"] for h in history], color="#1baf7a", label="val ROC-AUC")
    ax[1].set(title="Validation metrics (best checkpoint = epoch 70)", xlabel="epoch", ylim=(0.3, 0.9))
    ax[1].legend(loc="lower right")
    save(fig, "fig5_training_curves")


def fig_roc(yt, pt, t_val, t_test):
    fpr, tpr, thr = roc_curve(yt, pt)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(fpr, tpr, color=HEALTHY, label=f"GNN, test AUC = {roc_auc_score(yt, pt):.3f}")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="chance")
    for t, name, mk in [(t_val, f"t* from validation = {t_val:.3f}", "o"),
                        (t_test, f"t* from test (leaky) = {t_test:.3f}", "s")]:
        k = np.argmin(np.abs(thr - t))
        ax.plot(fpr[k], tpr[k], mk, ms=9, color=DISEASED if mk == "s" else INK,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC (N = 75)")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "fig6_roc_thresholds")


# ---------------------------------------------------------------------------
# Baselines re-run on the SAME split (flattened connectome)
# ---------------------------------------------------------------------------
def baselines():
    res = {}
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")).fit(X[itr], y[itr])
    s = svm.decision_function(X[ite])
    res["RBF-SVM (flattened)"] = (roc_auc_score(y[ite], s), accuracy_score(y[ite], svm.predict(X[ite])),
                                  f1_score(y[ite], svm.predict(X[ite]), average="macro"))
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000)).fit(X[itr], y[itr])
    p = lr.predict_proba(X[ite])[:, 1]
    res["L2 LogReg (flattened)"] = (roc_auc_score(y[ite], p), accuracy_score(y[ite], p >= 0.5),
                                    f1_score(y[ite], p >= 0.5, average="macro"))
    # one-scalar rule: threshold fit on train only
    f = np.abs(X[:, INTRA]).mean(1)
    thr = (f[itr][y[itr] == 0].mean() + f[itr][y[itr] == 1].mean()) / 2
    res["1 scalar: mean |z| intra-lobe"] = (roc_auc_score(y[ite], f[ite]), accuracy_score(y[ite], f[ite] >= thr),
                                           f1_score(y[ite], f[ite] >= thr, average="macro"))
    return res


def fig_compare(res, yt, pt, t_val):
    yh = (pt >= t_val).astype(int)
    res = dict(res)
    res["Ours: GCN+GATv2+SwiGLU"] = (roc_auc_score(yt, pt), accuracy_score(yt, yh), f1_score(yt, yh, average="macro"))
    names = list(res)
    aucs = [res[n][0] for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.grid(axis="y", visible=False)
    cols = [INK2, INK2, "#1baf7a", HEALTHY]
    ax.barh(names, aucs, color=cols, height=0.5)
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    for i, a in enumerate(aucs):
        ax.text(a + 0.01, i, f"{a:.3f}", va="center", color=INK)
    ax.set(xlim=(0, 1.1), xlabel="test ROC-AUC (same 75 subjects)", title="Re-run comparison on identical split")
    ax.invert_yaxis()
    save(fig, "fig7_model_comparison")
    return res


# ---------------------------------------------------------------------------
# Explainability: consensus edges vs the planted ground truth
# ---------------------------------------------------------------------------
def fig_explainer():
    ex = json.load(open(ROOT / "logs/biomarker_explanations.json"))
    cnt, n, intra, homo = Counter(), 0, 0, 0
    for s in ex:
        for e in s["top_edges"]:
            i, j = sorted((e["source_idx"], e["target_idx"]))
            cnt[(i, j)] += 1; n += 1
            intra += LOBE_OF[i] == LOBE_OF[j]
            homo += (i < 90 and i % 2 == 0 and j == i + 1)
    chance_intra = INTRA.mean()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    G = nx.Graph()
    G.add_nodes_from(range(116))
    for (i, j), c in cnt.most_common(40):
        G.add_edge(i, j, c=c, intra=LOBE_OF[i] == LOBE_OF[j])
    theta = np.linspace(0, 2 * np.pi, 116, endpoint=False)
    pos = {k: (np.cos(theta[k]), np.sin(theta[k])) for k in range(116)}
    ax[0].axis("off"); ax[0].set_aspect("equal")
    nx.draw_networkx_nodes(G, pos, ax=ax[0], node_size=18,
                           node_color=[LOBE_COLORS[LOBE_OF[k]] for k in range(116)])
    for u, v, d in G.edges(data=True):
        ax[0].plot(*zip(pos[u], pos[v]), color="#1baf7a" if d["intra"] else INK2,
                   lw=0.8 + 0.8 * d["c"], alpha=0.8)
    ax[0].plot([], [], color="#1baf7a", label="intra-lobe (planted signal)")
    ax[0].plot([], [], color=INK2, label="inter-lobe")
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2)
    ax[0].set_title("40 most frequent GNNExplainer edges (75 test subjects)")
    vals = [intra / n, chance_intra]
    ax[1].bar(["GNNExplainer\ntop-10 edges", "random edge\n(chance)"], vals,
              color=["#1baf7a", INK2], width=0.5)
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.005, f"{v:.1%}", ha="center", color=INK)
    ax[1].set(ylim=(0, 0.25), ylabel="fraction of edges that are intra-lobe",
              title="Explainer precision vs planted ground truth")
    save(fig, "fig8_explainer_vs_ground_truth")
    return intra / n, chance_intra, homo, n


if __name__ == "__main__":
    fig_eda()
    fig_signal()
    fig_diff_matrix()
    fig_brain_graph()
    yv, pv, yt, pt, t_val, t_test, hist = gnn_scores()
    fig_training(hist)
    fig_roc(yt, pt, t_val, t_test)
    res = fig_compare(baselines(), yt, pt, t_val)
    prec, chance, homo, n = fig_explainer()
    summary = {
        "gnn_val_auc": roc_auc_score(yv, pv), "gnn_test_auc": roc_auc_score(yt, pt),
        "t_val": t_val, "t_test_leaky": t_test,
        "gnn_test_acc_at_t_val": accuracy_score(yt, pt >= t_val),
        "gnn_test_acc_at_t_test_leaky": accuracy_score(yt, pt >= t_test),
        "comparison_auc_acc_f1": res,
        "explainer_intra_precision": prec, "chance_intra": chance,
        "explainer_homotopic_hits": homo, "explainer_edges_total": n,
    }
    json.dump(summary, open(OUT / "verified_numbers.json", "w"), indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
