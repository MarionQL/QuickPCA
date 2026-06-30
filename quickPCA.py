# quick_pca.py (ver 2.00)
# Essential Dynamics Analysis for MD trajectories.
# Runs inside PyMOL (drag-and-drop) or standalone via MDAnalysis.
# Supports PCA, TICA, and UMAP on single or multiple trajectories.
#
# Author: Gleb Novikov
# © The Visual Hub 2026
# For educational use only.
# If you use QuickPCA in your research, please cite this tool.
#
# INSTRUCTIONS:
#   Inside PyMOL : drag-and-drop this script into the PyMOL window.
#   Standalone   : python quick_pca.py [--help for all options]
#   Supported trajectory formats: .nc  .xtc  .trr  .dcd
#
# OUTPUT (per method: pca / tica / umap):
#   <prefix>_<METHOD>.png           — 4-panel report
#   <prefix>_<METHOD>_replicates.png — ellipse plot (multi-traj only)
#   <prefix>_<METHOD>_variance.csv  — EVR / kinetic variance / timescales
#   <prefix>_<METHOD>_loadings.csv  — eigenvectors (PCA / TICA only)
#   clustering/<method>/            — elbow plots, scatter, centroid PDBs
#
# DEPENDENCIES:
#   pip install numpy scikit-learn scipy matplotlib
#   pip install MDAnalysis              # standalone backend
#   pip install deeptime                # for TICA
#   pip install umap-learn              # for UMAP
 
import argparse
import glob
import os
import sys
import time
import platform
 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
 
# ── PyMOL backend detection ───────────────────────────────────────────────────
USE_PYMOL = False
try:
    from pymol import cmd
    USE_PYMOL = True
except ImportError:
    pass
 
# =============================================================================
# ⚙️  USER SETTINGS  — overridden by CLI flags when run standalone
# =============================================================================
 
# Atom selection
PCA_SEL        = "polymer and name CA"   # PyMOL syntax
PCA_SEL_MDA    = "name CA"              # MDAnalysis syntax
PCA_SEL_ACTIVE = PCA_SEL if USE_PYMOL else PCA_SEL_MDA
 
# Dimensionality reduction
METHODS   = ["pca"]       # any subset of: "pca", "tica", "umap"
PCA_MODE  = "count"       # "count" → use PCA_NCOMP  |  "variance" → use PCA_VAR
PCA_NCOMP = 10            # components to compute (count mode)
PCA_VAR   = 0.90          # cumulative variance threshold (variance mode)
TICA_LAG  = 10            # TICA lag time in strided frames
 
# Free-Energy Landscape
PCA_NBINS = 50            # histogram bins per axis
PCA_SIGMA = 1.0           # Gaussian smoothing σ (bins)
PCA_TEMP  = 300.0         # temperature (K)
 
# Trajectory sampling
MD_INTERVAL = 1           # use every Nth frame
 
# Output
OUTPUT_PREFIX = "Report"   # method suffix added automatically
EXPORT_CSV    = True
 
# Clustering
CLUSTER                   = True
CLUSTER_USE_UMAP          = False   # extra UMAP embedding inside clustering
CLUSTER_KMAX              = 12
CLUSTER_INERTIA_THRESHOLD = 0.15
CLUSTER_EXTRACT_CENTROIDS = True
CLUSTER_OUTDIR            = "clustering"
 
# Multi-trajectory (MDAnalysis only)
MULTI_TRAJ         = False
MULTI_TRAJ_PATTERN = "rep*.nc"      # glob to find replicate trajectories
REPLICATE_PLOT     = True
REPLICATE_ELLIPSE_LEVEL = 0.95      # confidence level for ellipses

# Out-of-sample projection (MDAnalysis standalone only)
PROJECT_TRAJ       = []             # secondary trajectory file(s) projected onto fitted axes
PROJECT_LABEL      = "Projected"    # label for secondary system in plots/CSVs
PROJECT_PLOT       = True           # save standalone overlay projection figure
 
# =============================================================================
# 🎨  PALETTE  (shared across all plots)
# =============================================================================
 
_PALETTE = [
    "steelblue", "coral", "teal", "darkorange",
    "mediumpurple", "seagreen", "crimson", "goldenrod",
    "slategray", "deeppink",
]
 
# =============================================================================
# 🔌  BACKEND HELPERS
# =============================================================================
 
def _kabsch(mobile, ref_pos, ref_com):
    """Pure-numpy Kabsch alignment. Used as PyMOL backend and fallback."""
    H        = (mobile - mobile.mean(0)).T @ (ref_pos - ref_com)
    U, _, Vt = np.linalg.svd(H)
    d        = np.sign(np.linalg.det(Vt.T @ U.T))
    R        = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (mobile - mobile.mean(0)) @ R.T + ref_com
 
# Use MDAnalysis C-level rotation_matrix when available (faster than numpy SVD)
try:
    from MDAnalysis.analysis.align import rotation_matrix as _mda_rotmat
    def _align(mobile, ref_pos, ref_com):
        R, _ = _mda_rotmat(mobile - mobile.mean(0), ref_pos - ref_com)
        return (mobile - mobile.mean(0)) @ R.T + ref_com
except ImportError:
    _align = _kabsch
 
 
def _extract_frames_pymol(obj_name, selection):
    """Extract Kabsch-aligned Cα frames from all PyMOL states."""
    n_states = cmd.count_states(obj_name)
    ref_pos = ref_com = None
    frames = []
    for state in range(1, n_states + 1):
        model  = cmd.get_model(f"({obj_name}) and ({selection})", state=state)
        coords = np.array([a.coord for a in model.atom], dtype=np.float64)
        if coords.size == 0:
            print(f"   ⚠️  State {state}: no atoms matched — skipping.")
            continue
        if ref_pos is None:
            ref_pos = coords.copy(); ref_com = ref_pos.mean(0)
        frames.append(_kabsch(coords, ref_pos, ref_com).ravel())
    return np.array(frames, dtype=np.float64), None   # None → no replicate ids
 
 
def _extract_frames_mda(universes, selection, ref_pos=None, ref_com=None, return_reference=False):
    """
    Extract alignment-corrected frames from one or more MDAnalysis Universes.

    If *ref_pos*/*ref_com* are supplied, every trajectory is aligned to that
    existing primary-system reference. This is required for valid out-of-sample
    projection onto previously fitted PCA/TICA/UMAP coordinates.
    """
    if not isinstance(universes, (list, tuple)):
        universes = [universes]

    n_per_rep = [len(range(0, len(u.trajectory), MD_INTERVAL)) for u in universes]
    n_total = int(sum(n_per_rep))

    ag0 = universes[0].select_atoms(selection)
    n_at = len(ag0)
    if n_at == 0:
        raise ValueError(f"Selection '{selection}' matched 0 atoms.")

    if ref_pos is not None:
        ref_pos = np.asarray(ref_pos, dtype=np.float64)
        if ref_pos.shape != (n_at, 3):
            raise ValueError(
                "Projection selection mismatch: secondary selection has "
                f"{n_at} atoms, but the primary reference has {ref_pos.shape[0]}."
            )
        ref_com = np.asarray(ref_com, dtype=np.float64)

    out = np.empty((n_total, n_at * 3), dtype=np.float64)
    rep_arr = np.empty(n_total, dtype=np.int32)
    row = 0

    for rep_idx, u in enumerate(universes, start=1):
        ag = u.select_atoms(selection)
        if len(ag) != n_at:
            raise ValueError(
                f"Rep {rep_idx}: atom count mismatch ({len(ag)} vs {n_at})."
            )

        for ts_idx in range(0, len(u.trajectory), MD_INTERVAL):
            u.trajectory[ts_idx]
            coords = ag.positions.astype(np.float64, copy=False)
            if ref_pos is None:
                ref_pos = coords.copy()
                ref_com = ref_pos.mean(0)
            out[row] = _align(coords, ref_pos, ref_com).ravel()
            rep_arr[row] = rep_idx
            row += 1

        print(
            f"   Rep {rep_idx} ({os.path.basename(str(u.filename))}): "
            f"{n_per_rep[rep_idx-1]} frames after stride {MD_INTERVAL}"
        )

    rep_ids = rep_arr if len(universes) > 1 else None
    if return_reference:
        return out, rep_ids, ref_pos, ref_com
    return out, rep_ids

def _extract_frames(system_or_list, selection, ref_pos=None, ref_com=None, return_reference=False):
    """Dispatch frame extraction to the correct backend."""
    if USE_PYMOL:
        if ref_pos is not None:
            raise NotImplementedError(
                "Out-of-sample projection is supported only by the MDAnalysis backend."
            )
        positions, rep_ids = _extract_frames_pymol(system_or_list, selection)
        if return_reference:
            return positions, rep_ids, None, None
        return positions, rep_ids
    return _extract_frames_mda(
        system_or_list, selection, ref_pos=ref_pos, ref_com=ref_com,
        return_reference=return_reference
    )

# =============================================================================
# 🔬  DIMENSIONALITY REDUCTION
# =============================================================================
 
def _cross_corr(evecs, weights, n_atoms):
    """Residue cross-correlation matrix from eigenvectors + eigenvalue weights."""
    evecs_3d   = evecs.reshape(len(evecs), n_atoms, 3)
    cov        = np.einsum('kia,kja,k->ij', evecs_3d, evecs_3d, np.abs(weights))
    var        = np.diag(cov)
    denom      = np.sqrt(np.outer(var, var))
    return np.where(denom > 0, cov / denom, 0.0).astype(np.float32)
 
 
def reduce_pca(positions, n_components=PCA_NCOMP):
    """
    PCA on the (n_frames × 3N) Cα coordinate matrix.
    Uses randomized SVD for count mode (15-17× faster); full SVD for variance mode.
    """
    from sklearn.decomposition import PCA as _PCA
 
    if PCA_MODE == "variance":
        n_comp, solver = PCA_VAR, "full"
    else:
        n_comp = min(n_components, min(positions.shape) - 1)
        solver = "randomized"
 
    centered = positions - positions.mean(0)
    model    = _PCA(n_components=n_comp, svd_solver=solver, random_state=42)
    proj     = model.fit_transform(centered)
 
    nc     = proj.shape[1]
    evr    = model.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    cc     = _cross_corr(model.components_, model.explained_variance_, positions.shape[1] // 3)
 
    print(f"   PC1={evr[0]*100:.1f}%  PC2={evr[1]*100:.1f}%  "
          f"top-{nc} cumul={cumvar[-1]*100:.1f}%")
 
    return dict(projections=proj, explained_variance_ratio=evr,
                cumulative_variance=cumvar, cross_correlation=cc,
                n_components=nc, eigenvectors=model.components_,
                model=model, fit_mean=positions.mean(0),
                bar_label="Explained Variance (%)", comp_label="PC")
 
 
def reduce_tica(frame_arrays, lag=None, n_components=PCA_NCOMP):
    """
    TICA via deeptime. *frame_arrays* is a list of (n_frames, n_features) arrays
    — one per trajectory — so TICA respects trajectory boundaries.
    """
    try:
        from deeptime.decomposition import TICA as _TICA
    except ImportError:
        print("❌  deeptime not found.  pip install deeptime")
        return None
 
    lag    = lag or TICA_LAG
    stacked = np.vstack(frame_arrays)
    n_feat  = stacked.shape[1]
 
    if PCA_MODE == "variance":
        # Fit with all components first, then trim to variance threshold
        n_init  = min(50, n_feat - 1)
        model   = _TICA(lagtime=lag, dim=n_init).fit(frame_arrays).fetch_model()
        sv2     = model.singular_values[:n_init] ** 2
        cumkv   = np.cumsum(sv2 / sv2.sum())
        n_comp  = int(np.searchsorted(cumkv, PCA_VAR) + 1)
        print(f"   TICA variance mode → {n_comp} ICs for {PCA_VAR:.0%} kinetic variance")
    else:
        n_comp = min(n_components, n_feat - 1)
 
    model  = _TICA(lagtime=lag, dim=n_comp).fit(frame_arrays).fetch_model()
    proj   = model.transform(stacked)
 
    # Kinetic variance fractions (VAMP-2 analog of EVR)
    sv2    = model.singular_values[:n_comp] ** 2
    evr    = sv2 / sv2.sum()
    cumvar = np.cumsum(evr)
 
    # Right singular vectors: shape (n_features, n_comp) → transpose to (n_comp, n_features)
    evecs  = model.singular_vectors_right[:, :n_comp].T
    cc     = _cross_corr(evecs, sv2, n_feat // 3)
    ts     = model.timescales(lagtime=lag)[:n_comp]
 
    print(f"   IC1={evr[0]*100:.1f}%  IC2={evr[1]*100:.1f}%  "
          f"top-{n_comp} cumul={cumvar[-1]*100:.1f}%")
    print(f"   Implied timescales (frames): {np.round(ts[:5], 1)}")
 
    return dict(projections=proj, explained_variance_ratio=evr,
                cumulative_variance=cumvar, cross_correlation=cc,
                n_components=n_comp, eigenvectors=evecs,
                model=model, timescales=ts, lag=lag,
                bar_label="Kinetic Variance (%)", comp_label="IC")
 
 
def reduce_umap(projections, n_components=2):
    """UMAP applied to existing projections (PCA or TICA output). No eigenvectors."""
    try:
        import umap as _umap
    except ImportError:
        print("❌  umap-learn not found.  pip install umap-learn")
        return None
    print("   Running UMAP …")
    model = _umap.UMAP(n_components=n_components, random_state=42)
    emb = model.fit_transform(projections)
    return dict(projections=emb, n_components=n_components,
                model=model, upstream_projections=projections,
                bar_label=None, comp_label="Dim")
 
 
def _reduce(method, positions, frame_arrays):
    """Dispatch to the correct reduction function and return a standardised result dict."""
    if method == "pca":
        return reduce_pca(positions)
    if method == "tica":
        return reduce_tica(frame_arrays)
    if method == "umap":
        # UMAP runs on top of PCA projections. Keep the upstream PCA model so
        # secondary trajectories can be projected through the same preprocessing.
        pca_r = reduce_pca(positions)
        r = reduce_umap(pca_r["projections"]) if pca_r else None
        if r is not None:
            r["upstream_model"] = pca_r.get("model")
            r["upstream_fit_mean"] = pca_r.get("fit_mean")
        return r
    print(f"❌  Unknown method: {method}")
    return None
 

def _project_positions(method, result, positions):
    """
    Project secondary Cartesian coordinates onto the fitted primary model.

    PCA is a true linear out-of-sample projection. TICA uses deeptime's fitted
    transform. UMAP uses umap-learn's approximate transform, so it should be
    interpreted as an embedding overlay rather than a physical eigenvector basis.
    """
    model = result.get("model")
    if model is None:
        raise ValueError(f"Cannot project onto {method.upper()}: fitted model was not stored.")

    if method == "pca":
        return model.transform(positions - result["fit_mean"])
    if method == "tica":
        return model.transform(positions)
    if method == "umap":
        if not hasattr(model, "transform"):
            raise ValueError("The installed umap-learn model does not support transform().")
        # Current UMAP is fit on PCA projections, so secondary coordinates must
        # first be transformed by the upstream PCA model stored in the result.
        upstream = result.get("upstream_model")
        upstream_mean = result.get("upstream_fit_mean")
        if upstream is None or upstream_mean is None:
            raise ValueError("UMAP projection requires the stored upstream PCA model.")
        secondary_pca = upstream.transform(positions - upstream_mean)
        return model.transform(secondary_pca)

    raise ValueError(f"Unknown projection method: {method}")


def _attach_projection(result, projected_positions, projected_rep_ids, method, label):
    """Add out-of-sample projection arrays to a method result dict."""
    proj = _project_positions(method, result, projected_positions)
    result["projected"] = {
        "label": label,
        "projections": np.asarray(proj, dtype=np.float64),
        "replicate_ids": projected_rep_ids,
    }
    return result


# =============================================================================
# 🌊  FREE-ENERGY LANDSCAPE
# =============================================================================
 
def compute_fel(result, temperature=PCA_TEMP, n_bins=PCA_NBINS, sigma=PCA_SIGMA):
    """Boltzmann inversion of PC1/PC2 (or IC1/IC2, Dim1/Dim2) density."""
    kBT  = 0.008314462 * temperature
    proj = result["projections"]
    x, y = proj[:, 0], proj[:, 1]
 
    pad_x = (x.max() - x.min()) * 0.20
    pad_y = (y.max() - y.min()) * 0.20
    rng   = [[x.min() - pad_x, x.max() + pad_x],
             [y.min() - pad_y, y.max() + pad_y]]
 
    hist, xe, ye = np.histogram2d(x, y, bins=n_bins, range=rng, density=True)
    hist_s = gaussian_filter(hist, sigma=sigma)
 
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(hist_s > 0, -kBT * np.log(hist_s), np.nan)
    F -= np.nanmin(F)
 
    return dict(F=F,
                xcenters=0.5*(xe[:-1]+xe[1:]), ycenters=0.5*(ye[:-1]+ye[1:]),
                xedges=xe, yedges=ye, x=x, y=y, kBT=kBT)
 
# =============================================================================
# 🔵  CLUSTERING
# =============================================================================
 
def _kmeans_elbow(projections, method_label):
    """KMeans elbow, saves plot, returns (labels, centres)."""
    from sklearn.cluster import KMeans
 
    ks       = list(range(2, CLUSTER_KMAX + 1))
    inertias = np.array([
        KMeans(n_clusters=k, random_state=42, n_init="auto").fit(projections).inertia_
        for k in ks])
    drops = -np.diff(inertias) / inertias[:-1]
 
    best_k = ks[0]
    for i, d in enumerate(drops):
        if d >= CLUSTER_INERTIA_THRESHOLD:
            best_k = ks[i + 1]
        else:
            break
    print(f"   🔵  {method_label} best k={best_k}  "
          f"(threshold {CLUSTER_INERTIA_THRESHOLD:.0%})")
 
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, inertias, "o-", color="steelblue", lw=2, ms=6, mec="navy", mew=0.8)
    for i in range(len(inertias) - 1):
        ax.text((ks[i]+ks[i+1])/2+0.1, (inertias[i]+inertias[i+1])/2,
                f"{drops[i]:.1%}", fontsize=8, ha="center", va="bottom")
    ax.axvline(best_k, color="coral", ls="--", lw=1.4, label=f"Best k={best_k}")
    ax.set_xlabel("k", fontsize=11, fontweight="bold")
    ax.set_ylabel("Inertia", fontsize=11, fontweight="bold")
    ax.set_title(f"Elbow — {method_label}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    os.makedirs(CLUSTER_OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(CLUSTER_OUTDIR, f"elbow_{method_label}.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
 
    km = KMeans(n_clusters=best_k, random_state=42, n_init="auto").fit(projections)
    return km.labels_, km.cluster_centers_
 
 
def _plot_cluster_embedding(emb2d, labels, centers2d, title, fname):
    """2-D scatter coloured by cluster. Matches report aesthetics."""
    unique_k = np.unique(labels)
    fig, ax  = plt.subplots(figsize=(8, 7))
    ax.scatter(emb2d[:, 0], emb2d[:, 1],
               c=[_PALETTE[i % len(_PALETTE)] for i in labels],
               edgecolors="k", linewidths=0.3, s=40, alpha=0.85, rasterized=True)
    for i, ctr in enumerate(centers2d):
        base  = np.array(mcolors.to_rgb(_PALETTE[i % len(_PALETTE)]))
        light = base + (1 - base) * 0.45
        ax.scatter(ctr[0], ctr[1], color=light, edgecolors="k",
                   s=220, marker="X", lw=1.5, zorder=10)
        ax.text(ctr[0], ctr[1], str(i), fontsize=9, fontweight="bold",
                ha="center", va="center", color="white", zorder=11)
    ax.legend(handles=[
        Line2D([0],[0], marker="o", color="w", label=f"Cluster {i}",
               markerfacecolor=_PALETTE[i%len(_PALETTE)], markeredgecolor="k", ms=8)
        for i in unique_k], title="Clusters", fontsize=9, title_fontsize=10)
    ax.set_xlabel("Dim 1", fontsize=11, fontweight="bold")
    ax.set_ylabel("Dim 2", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    os.makedirs(CLUSTER_OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(CLUSTER_OUTDIR, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊  {fname}")
 
 
def _extract_centroids(systems, embedding, labels, centers, rep_ids, label):
    """Save centroid PDB for each cluster. Handles single and multi-traj."""
    if not isinstance(systems, (list, tuple)):
        systems = [systems]
    outdir = os.path.join(CLUSTER_OUTDIR, f"centroids_{label}")
    os.makedirs(outdir, exist_ok=True)
 
    for cid, center in enumerate(centers):
        idx = np.where(labels == cid)[0]
        frame_idx = int(idx[np.argmin(np.linalg.norm(embedding[idx] - center, axis=1))])
        out_path  = os.path.join(outdir, f"cluster{cid}.pdb")
 
        if USE_PYMOL:
            tmp = f"_ctmp_{cid}"
            cmd.create(tmp, systems[0], source_state=frame_idx+1, target_state=1)
            cmd.save(out_path, tmp); cmd.delete(tmp)
        else:
            if rep_ids is not None:
                rep     = rep_ids[frame_idx]
                u       = systems[rep - 1]
                # frame_idx is global; compute local index within this replicate
                local   = int(np.where(rep_ids == rep)[0].tolist().index(frame_idx))
                u.trajectory[local * MD_INTERVAL]
            else:
                systems[0].trajectory[frame_idx * MD_INTERVAL]
                u = systems[0]
            u.atoms.write(out_path)
        print(f"   💾  Cluster {cid} centroid (frame {frame_idx}) → {out_path}")
 
 
def run_clustering(result, systems, method_label):
    """
    KMeans clustering (+ optional UMAP re-embedding) on *result['projections']*.
    Single entry point; works for PCA, TICA, and UMAP results.
    """
    if not CLUSTER:
        return
    proj    = result["projections"]
    rep_ids = result.get("replicate_ids")
    os.makedirs(CLUSTER_OUTDIR, exist_ok=True)
    print(f"\n🔵  Clustering ({method_label}) …")

    clustering_results = {}
 
    embeddings = {method_label: proj}
    if CLUSTER_USE_UMAP:
        try:
            import umap as _umap
            embeddings[f"{method_label}_UMAP"] = (
                _umap.UMAP(random_state=42).fit_transform(proj))
        except ImportError:
            print("   ⚠️  umap-learn not found — skipping UMAP.  pip install umap-learn")
 
    for emb_label, emb in embeddings.items():
        labels, centers = _kmeans_elbow(emb, emb_label)
        _plot_cluster_embedding(emb[:, :2], labels, centers[:, :2],
                                title=f"Clusters — {emb_label}",
                                fname=f"clusters_{emb_label}.png")
        csv_path = os.path.join(CLUSTER_OUTDIR, f"labels_{emb_label}.csv")
        np.savetxt(csv_path, np.column_stack([np.arange(len(labels)), labels]),
                   delimiter=",", header="frame,cluster", comments="", fmt="%d")
        print(f"   🗂️  Labels → {csv_path}")
        clustering_results[emb_label] = {
            "embedding": emb,
            "labels": labels,
            "centers": centers,
        }

        if CLUSTER_EXTRACT_CENTROIDS:
            _extract_centroids(systems, emb, labels, centers, rep_ids, emb_label)
    return clustering_results
 
# =============================================================================
# 📊  REPORT FIGURE  — unified 4-panel layout for PCA / TICA / UMAP
# =============================================================================
 
def _panel_fel(ax, fig, fel, result, method):
    """Panel 1: Free-Energy Landscape (identical for all methods)."""
    comp_label = result.get("comp_label", "Dim")
    evr        = result.get("explained_variance_ratio")
    F    = fel["F"]
    x, y = fel["x"], fel["y"]
 
    F_plot = np.where(np.isnan(F), np.nanmax(F), F)
    XX, YY = np.meshgrid(fel["xcenters"], fel["ycenters"])
    levels = np.linspace(0, np.nanpercentile(F, 97), 30)
 
    cf = ax.contourf(XX, YY, F_plot.T, levels=levels, cmap="RdYlBu_r", extend="max")
    ax.contour(XX, YY, F_plot.T, levels=levels[::5],
               colors="k", linewidths=0.4, alpha=0.5)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Free Energy (kJ mol⁻¹)", fontsize=10)
    F_max = np.nanpercentile(F, 97)
    cb.set_ticks(range(0, int(F_max) + max(1, int(round(F_max/6))),
                       max(1, int(round(F_max/6)))))
 
    ax.plot(x, y, color="white", lw=0.25, alpha=0.3, rasterized=True)
    ax.scatter(x[0],  y[0],  c="lime", s=130, marker="*",
               zorder=5, edgecolors="k", lw=0.7, label="Start")
    ax.scatter(x[-1], y[-1], c="red",  s=130, marker="*",
               zorder=5, edgecolors="k", lw=0.7, label="End")
 
    xlabel = f"{comp_label}1 ({evr[0]*100:.1f}%)" if evr is not None else f"{comp_label}1"
    ylabel = f"{comp_label}2 ({evr[1]*100:.1f}%)" if evr is not None else f"{comp_label}2"
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(f"Free-Energy Landscape  ({method.upper()}, T={PCA_TEMP:.0f} K)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(fel["xedges"][0], fel["xedges"][-1])
    ax.set_ylim(fel["yedges"][0], fel["yedges"][-1])
    ax.legend(fontsize=10, loc="upper right", frameon=True)
 
 
def _panel_corr_or_time(ax, fig, result, method):
    """
    Panel 2 (top-right):
      PCA / TICA → residue cross-correlation matrix
      UMAP       → trajectory-progression scatter (time-coloured)
    """
    if "cross_correlation" in result:
        cc = result["cross_correlation"]
        im = ax.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1,
                       aspect="auto", origin="lower", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cross-correlation")
        ax.set_xlabel("Residue index", fontsize=11, fontweight="bold")
        ax.set_ylabel("Residue index", fontsize=11, fontweight="bold")
        ax.set_title("Residue Cross-Correlation Matrix",
                     fontsize=12, fontweight="bold")
    else:
        # UMAP: colour by frame number to show trajectory sampling
        proj = result["projections"]
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(proj)),
                        cmap="viridis", s=15, alpha=0.75, rasterized=True)
        fig.colorbar(sc, ax=ax, label="Frame index")
        ax.set_xlabel("Dim 1", fontsize=11, fontweight="bold")
        ax.set_ylabel("Dim 2", fontsize=11, fontweight="bold")
        ax.set_title("UMAP — Trajectory Progression",
                     fontsize=12, fontweight="bold")
 
 
def _panel_variance_or_timescales(ax, result):
    """
    Panel 3 (bottom-left):
      PCA  → explained-variance bar chart
      TICA → implied timescales bar chart
      UMAP → text note (non-linear, no EVR)
    """
    nc      = result["n_components"]
    n_show  = min(nc, 10)
    bar_lbl = result.get("bar_label")
    evr     = result.get("explained_variance_ratio")
 
    if "timescales" in result:
        # TICA implied timescales — clip negatives (unphysical; arise from
        # random/short data where processes are faster than the lag time)
        ts      = result["timescales"]
        ts_plot = np.abs(ts[:min(len(ts), n_show)])
        n_neg   = int((ts[:len(ts_plot)] < 0).sum())
        x       = range(1, len(ts_plot) + 1)
        bars    = ax.bar(x, ts_plot, color="teal", alpha=0.85,
                         edgecolor="darkcyan", lw=0.6)
        for bar, v in zip(bars, ts_plot):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        if n_neg:
            ax.set_title(f"TICA Implied Timescales  (lag={result.get('lag', TICA_LAG)}) "
                         f"— {n_neg} IC(s) negative (|·| shown)",
                         fontsize=11, fontweight="bold")
        else:
            ax.set_title(f"TICA Implied Timescales  (lag={result.get('lag', TICA_LAG)})",
                         fontsize=12, fontweight="bold")
        ax.set_xlabel("IC", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"Timescale (×{MD_INTERVAL} frames)",
                      fontsize=10, fontweight="bold")
        ax.set_axisbelow(True)
 
    elif evr is not None:
        # PCA explained variance
        x_ticks = range(1, n_show + 1)
        bars = ax.bar(x_ticks, evr[:n_show]*100, color="steelblue",
                      alpha=0.85, edgecolor="navy", lw=0.6)
        for bar, pct in zip(bars, evr[:n_show]*100):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{pct:.1f}%", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
        ax2 = ax.twinx()
        ax2.plot(x_ticks, np.cumsum(evr[:n_show])*100, "o--", color="coral",
                 lw=1.8, ms=5, label="Cumulative")
        ax2.set_ylabel("Cumulative (%)", fontsize=10, color="coral")
        ax2.tick_params(axis="y", labelcolor="coral")
        ax2.set_ylim(0, 105)
        for thresh in (80, 90):
            ax2.axhline(thresh, ls=":", color="gray", alpha=0.6, lw=1.0)
        ax2.legend(loc="center right", fontsize=9)
        ax.set_xlabel("Principal Component", fontsize=11, fontweight="bold")
        ax.set_ylabel(bar_lbl or "Explained Variance (%)", fontsize=11, fontweight="bold")
        ax.set_title(f"First {n_show} PCs — Explained Variance",
                     fontsize=12, fontweight="bold")
        ax.set_xticks(list(x_ticks))
        ax.set_ylim(0, 105)
        ax.set_axisbelow(True)
 
    else:
        return False
 
 
def _panel_kde(ax, result):
    """Panel 4: 1-D projection histograms + KDE (identical for all methods)."""
    proj       = result["projections"]
    comp_label = result.get("comp_label", "Dim")
    evr        = result.get("explained_variance_ratio")
 
    for i, (color, name) in enumerate(zip(["teal", "darkorange"],
                                          [f"{comp_label}1", f"{comp_label}2"])):
        comp = proj[:, i]
        pct_str = f" ({evr[i]*100:.1f}%)" if evr is not None else ""
        ax.hist(comp, bins=60, color=color, alpha=0.45, edgecolor="k",
                lw=0.3, density=True, label=f"{name}{pct_str}")
        xr = np.linspace(comp.min(), comp.max(), 300)
        ax.plot(xr, gaussian_kde(comp)(xr), color=color, lw=2.0)
        ax.axvline(comp.mean(), color=color, ls="--", lw=1.2)
 
    ax.set_xlabel("Projection value", fontsize=11, fontweight="bold")
    ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.set_title("Projection Distributions", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
 
def _panel_top_loadings(ax, result):
    """
    Show top 10 atom/residue loading magnitudes for the first two components.
    PCA -> PC1/PC2
    TICA -> IC1/IC2
    """
    if "eigenvectors" not in result:
        return False

    evecs = result["eigenvectors"]
    comp_label = result.get("comp_label", "Comp")
    n_atoms = evecs.shape[1] // 3
    n_comp = min(2, evecs.shape[0])

    top_sets = []

    for i in range(n_comp):
        vec3 = evecs[i].reshape(n_atoms, 3)
        mag = np.linalg.norm(vec3, axis=1)

        top_idx = np.argsort(mag)[-10:][::-1]
        top_sets.append((i + 1, top_idx, mag[top_idx]))

    # combine top PC1 and top PC2 indices so the same x-axis works
    combined_idx = np.unique(np.concatenate([x[1] for x in top_sets]))
    combined_idx = combined_idx[np.argsort(combined_idx)]

    x = np.arange(len(combined_idx))
    width = 0.38

    for j, (comp_num, top_idx, vals) in enumerate(top_sets):
        mag_lookup = dict(zip(top_idx, vals))
        y = np.array([mag_lookup.get(idx, 0.0) for idx in combined_idx])

        offset = (j - 0.5) * width if n_comp == 2 else 0
        ax.bar(x + offset, y, width=width, alpha=0.85,
               label=f"{comp_label}{comp_num}")

    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in combined_idx], rotation=45, ha="right")
    ax.set_xlabel("Atom / residue index", fontsize=11, fontweight="bold")
    ax.set_ylabel("Loading magnitude", fontsize=11, fontweight="bold")
    ax.set_title(f"Top 10 {comp_label}1/{comp_label}2 Loadings",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_axisbelow(True)

    return True

def _panel_clusters(ax, result, method):
    """Cluster scatter panel inside the main report."""
    clustering = result.get("clustering")
    if not clustering:
        return False

    # Use the main method embedding, not optional nested UMAP, unless that is all that exists
    preferred = method.upper()
    key = preferred if preferred in clustering else next(iter(clustering))

    data = clustering[key]
    emb = data["embedding"]
    labels = data["labels"]
    centers = data["centers"]

    ax.scatter(
        emb[:, 0], emb[:, 1],
        c=[_PALETTE[i % len(_PALETTE)] for i in labels],
        edgecolors="k", linewidths=0.25, s=22, alpha=0.75,
        rasterized=True
    )

    for i, ctr in enumerate(centers):
        ax.scatter(
            ctr[0], ctr[1],
            color=_PALETTE[i % len(_PALETTE)],
            edgecolors="k", s=160, marker="X", lw=1.2, zorder=10
        )
        ax.text(ctr[0], ctr[1], str(i), fontsize=8, fontweight="bold",
                ha="center", va="center", color="white", zorder=11)

    ax.set_xlabel("Dim 1", fontsize=11, fontweight="bold")
    ax.set_ylabel("Dim 2", fontsize=11, fontweight="bold")
    ax.set_title(f"Clusters — {key}", fontsize=12, fontweight="bold")

    return True

def _panel_replicates(ax, result, method):
    """Replicate scatter + confidence ellipses inside the main report."""
    rep_ids = result.get("replicate_ids")
    if rep_ids is None:
        return False

    proj = result["projections"]
    evr = result.get("explained_variance_ratio")
    comp_lbl = result.get("comp_label", "Dim")

    for i, rep in enumerate(np.unique(rep_ids)):
        mask = rep_ids == rep
        color = _PALETTE[i % len(_PALETTE)]

        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            s=20, alpha=0.65, color=color,
            edgecolors="k", lw=0.25, rasterized=True,
            label=f"Rep {rep}"
        )

        _confidence_ellipse(
            ax, proj[mask, 0], proj[mask, 1],
            color=color, level=REPLICATE_ELLIPSE_LEVEL
        )

    xlabel = f"{comp_lbl}1 ({evr[0]*100:.1f}%)" if evr is not None else f"{comp_lbl}1"
    ylabel = f"{comp_lbl}2 ({evr[1]*100:.1f}%)" if evr is not None else f"{comp_lbl}2"

    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(f"{method.upper()} — Replicate Ellipses",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    return True

def _axis_labels(result):
    """Consistent axis labels for the first two components/dimensions."""
    comp_lbl = result.get("comp_label", "Dim")
    evr = result.get("explained_variance_ratio")
    xlabel = f"{comp_lbl}1 ({evr[0]*100:.1f}%)" if evr is not None else f"{comp_lbl}1"
    ylabel = f"{comp_lbl}2 ({evr[1]*100:.1f}%)" if evr is not None else f"{comp_lbl}2"
    return xlabel, ylabel


def _panel_projection_overlay(ax, result, method):
    """Overlay primary-system embedding with secondary out-of-sample projection."""
    projected = result.get("projected")
    if not projected:
        return False

    primary = result["projections"]
    secondary = projected["projections"]
    label = projected.get("label", "Projected")

    ax.scatter(primary[:, 0], primary[:, 1], s=18, alpha=0.38,
               color="slategray", edgecolors="none", rasterized=True,
               label=f"Fit system (n={len(primary)})")
    ax.scatter(secondary[:, 0], secondary[:, 1], s=24, alpha=0.78,
               color="crimson", edgecolors="k", linewidths=0.25,
               rasterized=True, label=f"{label} (n={len(secondary)})")

    # Show the center shift, which is often the most interpretable summary.
    p_mu = primary[:, :2].mean(axis=0)
    s_mu = secondary[:, :2].mean(axis=0)
    ax.scatter(p_mu[0], p_mu[1], marker="X", s=150, color="white",
               edgecolors="k", linewidths=1.1, zorder=5)
    ax.scatter(s_mu[0], s_mu[1], marker="X", s=170, color="gold",
               edgecolors="k", linewidths=1.1, zorder=6)
    ax.plot([p_mu[0], s_mu[0]], [p_mu[1], s_mu[1]], "k--", lw=1.2, alpha=0.75)

    xlabel, ylabel = _axis_labels(result)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(f"Out-of-Sample Projection onto {method.upper()} Axes",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=True)
    ax.set_axisbelow(True)
    return True


def plot_projection_overlay(result, method, output):
    """Standalone primary-vs-projected overlay plot."""
    if not result.get("projected"):
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    _panel_projection_overlay(ax, result, method)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊  Projection overlay → {output}")


def plot_report(result, method, output):
    """
    Universal 4-panel report. Works for PCA, TICA, and UMAP.
    Dispatches each panel based on keys present in *result*.
    """
    fel = compute_fel(result)
    panel_fns = []
    panel_fns.append(lambda ax, fig: _panel_fel(ax, fig, fel, result, method))

    if result.get("projected"):
        panel_fns.append(lambda ax, fig: _panel_projection_overlay(ax, result, method))

    if result.get("replicate_ids") is not None:
        panel_fns.append(lambda ax, fig: _panel_replicates(ax, result, method))

    if result.get("clustering"):
        panel_fns.append(lambda ax, fig: _panel_clusters(ax, result, method))

    if "eigenvectors" in result:
        panel_fns.append(lambda ax, fig: _panel_top_loadings(ax, result))

    # Keep correlation matrix only for PCA
    if method == "pca" and "cross_correlation" in result:
        panel_fns.append(lambda ax, fig: _panel_corr_or_time(ax, fig, result, method))

    if ("timescales" in result) or (result.get("explained_variance_ratio") is not None):
        panel_fns.append(lambda ax, fig: _panel_variance_or_timescales(ax, result))

    panel_fns.append(lambda ax, fig: _panel_kde(ax, result))

    n_panels = len(panel_fns)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig = plt.figure(figsize=(16, 6.5 * nrows))
    fig.suptitle(f"Essential Dynamics  —  {method.upper()} Report",
                 fontsize=15, fontweight="bold")
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.32, wspace=0.35,
                  top=0.94, bottom=0.06, left=0.07, right=0.97)
 
    for i, panel_fn in enumerate(panel_fns):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        panel_fn(ax, fig)
 
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"👑  {method.upper()} report saved → {output}")
    plt.close(fig)
    _open_file(output)
 
# =============================================================================
# 🔵  REPLICATE ELLIPSE PLOT
# =============================================================================
 
def _confidence_ellipse(ax, x, y, color, level=0.95):
    """Covariance ellipse for 2-D Gaussian data at given confidence level."""
    chi2_lut = {0.68: 2.279, 0.90: 4.605, 0.95: 5.991, 0.99: 9.210}
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(np.maximum(cov, 0))
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h  = 2 * np.sqrt(vals * chi2_lut.get(level, 5.991))
    ax.add_patch(Ellipse(xy=(x.mean(), y.mean()), width=w, height=h,
                         angle=angle, facecolor=color, edgecolor=color,
                         alpha=0.13, lw=1.5))
 
 
def plot_replicate_embedding(result, method, output):
    """
    Scatter + confidence ellipses coloured by replicate.
    Requires 'replicate_ids' in *result*. Works for any method.
    """
    rep_ids = result.get("replicate_ids")
    if rep_ids is None:
        print("ℹ️  No replicate metadata — skipping replicate plot.")
        return
    proj    = result["projections"]
    evr     = result.get("explained_variance_ratio")
    comp_lbl = result.get("comp_label", "Dim")
 
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, rep in enumerate(np.unique(rep_ids)):
        mask  = rep_ids == rep
        color = _PALETTE[i % len(_PALETTE)]
        ax.scatter(proj[mask, 0], proj[mask, 1], s=22, alpha=0.65, color=color,
                   edgecolors="k", lw=0.25, rasterized=True,
                   label=f"Rep {rep}  (n={mask.sum()})")
        _confidence_ellipse(ax, proj[mask, 0], proj[mask, 1],
                            color=color, level=REPLICATE_ELLIPSE_LEVEL)
 
    xlabel = (f"{comp_lbl}1 ({evr[0]*100:.1f}%)" if evr is not None
              else f"{comp_lbl}1")
    ylabel = (f"{comp_lbl}2 ({evr[1]*100:.1f}%)" if evr is not None
              else f"{comp_lbl}2")
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(f"{method.upper()} — Projection by Replicate  "
                 f"({REPLICATE_ELLIPSE_LEVEL:.0%} ellipses)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=True)
    ax.set_axisbelow(True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊  Replicate plot → {output}")
 
# =============================================================================
# 🗂️  CSV EXPORT
# =============================================================================
 
def export_csv(result, method, prefix):
    """Save variance / timescale and loadings CSVs for the given method."""
    evr    = result.get("explained_variance_ratio")
    cumvar = result.get("cumulative_variance")
    nc     = result["n_components"]
    ts     = result.get("timescales")
 
    if evr is not None:
        cols = [np.arange(1, nc+1), evr*100, cumvar*100]
        hdrs = "component,variance_pct,cumulative_pct"
        if ts is not None:
            cols.append(ts); hdrs += ",timescale_frames"
        np.savetxt(f"{prefix}_{method}_variance.csv",
                   np.column_stack(cols), delimiter=",",
                   header=hdrs, comments="",
                   fmt=["%d"] + ["%.4f"] * (len(cols) - 1))
 
    if "eigenvectors" in result:
        evecs = result["eigenvectors"][:nc]
        header = "component," + ",".join(f"feature_{i+1}"
                                         for i in range(evecs.shape[1]))
        rows = np.column_stack([np.arange(1, nc+1)[:, None], evecs])
        np.savetxt(f"{prefix}_{method}_loadings.csv", rows,
                   delimiter=",", header=header, comments="",
                   fmt=["%d"] + ["%.6f"]*evecs.shape[1])

    if result.get("projected"):
        projected = result["projected"]
        proj = projected["projections"][:, :min(2, projected["projections"].shape[1])]
        rep_ids = projected.get("replicate_ids")
        if rep_ids is None:
            rep_ids = np.ones(len(proj), dtype=np.int32)
        rows = np.column_stack([np.arange(len(proj)), rep_ids, proj])
        header = "frame,project_replicate," + ",".join(
            f"{result.get('comp_label', 'Dim')}{i+1}" for i in range(proj.shape[1])
        )
        np.savetxt(f"{prefix}_{method}_projected.csv", rows,
                   delimiter=",", header=header, comments="",
                   fmt=["%d", "%d"] + ["%.6f"] * proj.shape[1])
 
    print(f"   🗂️  CSV → {prefix}_{method}_variance.csv  "
          f"(+ _loadings.csv)" if "eigenvectors" in result else "")
 
# =============================================================================
# 🚀  MAIN PIPELINE
# =============================================================================
 
def _open_file(path):
    """Open output file in the default viewer (cross-platform)."""
    time.sleep(0.3)
    if platform.system() == "Darwin":
        os.system(f"open {path}")
    elif platform.system() == "Windows":
        os.system(f"start {path}")
    else:
        os.system(f"xdg-open {path}")
 
 
def _run_pipeline(systems, selection, methods, prefix, outdir=".", projected_systems=None, projected_label=PROJECT_LABEL):
    """
    Core analysis loop. Called by both main() (PyMOL / auto-discover) and
    main_cli() (argparse). Accepts single system or list of systems.
    """
    if not isinstance(systems, (list, tuple)):
        systems = [systems]
 
    # ── Extract frames once, reuse for all methods ──────────────────────────
    print(f"\n📐  Extracting frames  (selection: '{selection}') …")
    positions, rep_ids, ref_pos, ref_com = _extract_frames(
        systems, selection, return_reference=True
    )
    print(f"   Total: {len(positions)} frames × {positions.shape[1]} features")

    projected_positions = projected_rep_ids = None
    if projected_systems is not None:
        print(f"\n📐  Extracting projected frames  (label: '{projected_label}') …")
        projected_positions, projected_rep_ids = _extract_frames(
            projected_systems, selection, ref_pos=ref_pos, ref_com=ref_com
        )
        if projected_positions.shape[1] != positions.shape[1]:
            raise ValueError(
                "Projected trajectories do not have the same selected feature count "
                f"({projected_positions.shape[1]} vs {positions.shape[1]})."
            )
        print(
            f"   Projected total: {len(projected_positions)} frames × "
            f"{projected_positions.shape[1]} features"
        )
    if len(positions) < 3:
        print("❌  Not enough frames for analysis."); return
 
    # Split per-replicate for TICA (must not concatenate across boundaries)
    if rep_ids is not None:
        frame_arrays = [positions[rep_ids == r] for r in np.unique(rep_ids)]
    else:
        frame_arrays = [positions]
 
    # ── Run each requested method ────────────────────────────────────────────
    for method in methods:
        print(f"\n💠  {method.upper()} …")
        result = _reduce(method, positions, frame_arrays)
        if result is None:
            continue
        if rep_ids is not None:
            result["replicate_ids"] = rep_ids

        if projected_positions is not None:
            _attach_projection(
                result, projected_positions, projected_rep_ids,
                method=method, label=projected_label
            )
 
        out_report   = os.path.join(outdir, f"{prefix}_{method.upper()}.png")
        out_replicates = os.path.join(outdir, f"{prefix}_{method.upper()}_replicates.png")
        out_projected = os.path.join(outdir, f"{prefix}_{method.upper()}_projected.png")
 
        # Run clustering before report so cluster panels can be included
        if CLUSTER:
            result["clustering"] = run_clustering(result, systems, method_label=method.upper())

        plot_report(result, method=method, output=out_report)

        # Still save standalone diagnostic plots too
        if PROJECT_PLOT and result.get("projected"):
            plot_projection_overlay(result, method, output=out_projected)

        if REPLICATE_PLOT and rep_ids is not None:
            plot_replicate_embedding(result, method, output=out_replicates)

        if EXPORT_CSV:
            export_csv(result, method, os.path.join(outdir, prefix))
 
 
def main():
    """
    Entry point when run from PyMOL (drag-and-drop) or standalone without args.
    Auto-discovers topology and trajectory in the current directory.
    """
    start = time.time()
    print(f"🔌  Backend: {'PyMOL' if USE_PYMOL else 'MDAnalysis'}")
 
    traj = next((f for ext in ("*.nc","*.xtc","*.trr","*.dcd")
                 for f in glob.glob(ext)), None)
 
    if USE_PYMOL:
        objs = cmd.get_names("objects")
        if not objs:
            print("❌  No objects loaded in PyMOL."); return
        target = objs[0]
        print(f"✨  Target object: {target}")
        if traj:
            print(f"💫  Loading trajectory: {traj}")
            cmd.load_traj(traj, target, interval=MD_INTERVAL)
        else:
            print("ℹ️  No trajectory file found — using states already in PyMOL.")
        systems = target
 
    else:
        try:
            import MDAnalysis as mda
        except ImportError:
            print("❌  MDAnalysis not available.  pip install MDAnalysis"); return
 
        top = next(iter(glob.glob("*.pdb")), None)
        if not top:
            print("❌  No .pdb topology found in this folder."); return
        print(f"✨  Topology: {top}")
 
        if MULTI_TRAJ:
            trajs = sorted(glob.glob(MULTI_TRAJ_PATTERN))
            if not trajs:
                print(f"❌  No files matching '{MULTI_TRAJ_PATTERN}' found."); return
            print(f"💫  Multi-traj: {trajs}")
            systems = [mda.Universe(top, t) for t in trajs]
        elif traj:
            print(f"💫  Trajectory: {traj}")
            systems = mda.Universe(top, traj)
        else:
            print("ℹ️  No trajectory — using single topology frame.")
            systems = mda.Universe(top)
 
    _run_pipeline(systems, PCA_SEL_ACTIVE, METHODS, OUTPUT_PREFIX)
 
    elapsed = time.time() - start
    h, r = divmod(elapsed, 3600); m, s = divmod(r, 60)
    print(f"\n🕰️  Total: {int(h)}h {int(m)}m {int(s)}s")
 
 
def main_cli():
    """CLI entry point — used when the script is invoked directly (not from PyMOL)."""
    # Declare globals first so we can override them from CLI args below
    global PCA_NCOMP, PCA_VAR, PCA_MODE, TICA_LAG, MD_INTERVAL, PCA_TEMP
    global EXPORT_CSV, CLUSTER, CLUSTER_KMAX, CLUSTER_EXTRACT_CENTROIDS
    global REPLICATE_ELLIPSE_LEVEL, PROJECT_LABEL, PROJECT_PLOT
 
    p = argparse.ArgumentParser(
        prog="quick_pca.py",
        description="QuickPCA v2.00 — Essential Dynamics (PCA / TICA / UMAP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-t", "--topology",   required=False,
                   help="Topology file (.pdb). Auto-detected if omitted.")
    p.add_argument("-r", "--trajectory", nargs="+",
                   help="Trajectory file(s). Multiple = multi-traj mode.")
    p.add_argument("--project-trajectory", nargs="+", default=None,
                   help=("Secondary trajectory file(s) to project onto the fitted "
                         "primary PCA/TICA/UMAP coordinates. Must use the same topology "
                         "and atom selection/order as --trajectory."))
    p.add_argument("--project-label", default=PROJECT_LABEL,
                   help="Legend/CSV label for the projected secondary system.")
    p.add_argument("--no-project-plot", action="store_true",
                   help="Skip standalone primary-vs-projected overlay plot.")
    p.add_argument("-s", "--selection",  default=PCA_SEL_MDA,
                   help="MDAnalysis atom selection string.")
    p.add_argument("-m", "--method",     nargs="+",
                   default=METHODS, choices=["pca","tica","umap"],
                   metavar="METHOD",
                   help="Reduction method(s): pca, tica, umap.")
    p.add_argument("--ncomp",    type=int,   default=PCA_NCOMP,
                   help="Number of components (count mode).")
    p.add_argument("--var",      type=float, default=PCA_VAR,
                   help="Cumulative variance threshold (variance mode).")
    p.add_argument("--mode",     default=PCA_MODE, choices=["count","variance"],
                   help="Component-count mode.")
    p.add_argument("--lag",      type=int,   default=TICA_LAG,
                   help="TICA lag time (in strided frames).")
    p.add_argument("--interval", type=int,   default=MD_INTERVAL,
                   help="Frame stride (every Nth frame).")
    p.add_argument("--temp",     type=float, default=PCA_TEMP,
                   help="Temperature (K) for FEL Boltzmann inversion.")
    p.add_argument("--prefix",   default=OUTPUT_PREFIX,
                   help="Output filename prefix.")
    p.add_argument("--outdir",   default=".",
                   help="Output directory.")
    p.add_argument("--no-csv",   action="store_true",
                   help="Skip CSV export.")
    p.add_argument("--no-cluster", action="store_true",
                   help="Skip clustering.")
    p.add_argument("--kmax",     type=int,   default=CLUSTER_KMAX,
                   help="Max k for KMeans elbow.")
    p.add_argument("--no-centroids", action="store_true",
                   help="Skip centroid PDB extraction.")
    p.add_argument("--ellipse-level", type=float, default=REPLICATE_ELLIPSE_LEVEL,
                   help="Confidence level for replicate ellipses.")
 
    args = p.parse_args()
 
    # Apply CLI args to globals so all functions pick them up
    PCA_NCOMP                = args.ncomp
    PCA_VAR                  = args.var
    PCA_MODE                 = args.mode
    TICA_LAG                 = args.lag
    MD_INTERVAL              = args.interval
    PCA_TEMP                 = args.temp
    EXPORT_CSV               = not args.no_csv
    CLUSTER                  = not args.no_cluster
    CLUSTER_KMAX             = args.kmax
    CLUSTER_EXTRACT_CENTROIDS = not args.no_centroids
    REPLICATE_ELLIPSE_LEVEL  = args.ellipse_level
    PROJECT_LABEL            = args.project_label
    PROJECT_PLOT             = not args.no_project_plot
 
    try:
        import MDAnalysis as mda
    except ImportError:
        print("❌  MDAnalysis not available.  pip install MDAnalysis"); sys.exit(1)
 
    # Topology
    top = args.topology or next(iter(glob.glob("*.pdb")), None)
    if not top:
        print("❌  No topology file found. Use --topology or place a .pdb here.")
        sys.exit(1)
 
    # Trajectories
    if args.trajectory:
        trajs   = args.trajectory
        systems = [mda.Universe(top, t) for t in trajs] if len(trajs) > 1 \
                  else mda.Universe(top, trajs[0])
    else:
        traj = next((f for ext in ("*.nc","*.xtc","*.trr","*.dcd")
                     for f in glob.glob(ext)), None)
        systems = mda.Universe(top, traj) if traj else mda.Universe(top)
 
    projected_systems = None
    if args.project_trajectory:
        projected_systems = ([mda.Universe(top, t) for t in args.project_trajectory]
                             if len(args.project_trajectory) > 1
                             else mda.Universe(top, args.project_trajectory[0]))

    os.makedirs(args.outdir, exist_ok=True)
    start = time.time()
    print(f"🔌  Backend: MDAnalysis")
    _run_pipeline(
        systems, args.selection, args.method, args.prefix, args.outdir,
        projected_systems=projected_systems, projected_label=args.project_label
    )
    elapsed = time.time() - start
    h, r = divmod(elapsed, 3600); m, s = divmod(r, 60)
    print(f"\n🕰️  Total: {int(h)}h {int(m)}m {int(s)}s")
 
 
# =============================================================================
# ENTRYPOINT
# =============================================================================
 
if __name__ == "__main__":
    main_cli()        # standalone: python quick_pca.py [options]
elif USE_PYMOL:
    main()            # PyMOL drag-and-drop
# else: imported as module — do nothing
