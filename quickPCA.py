#!/usr/bin/env python3
# =============================================================================
# quick_pca.py  —  Essential Dynamics Analysis for MD trajectories (v3.0-yaml)
# =============================================================================
#
# Inspired by:
# Author: Gleb Novikov
# © The Visual Hub 2026
#
# WHAT THIS DOES
# ---------------
# Runs dimensionality reduction (PCA / TICA / UMAP) on an MD trajectory (or a
# set of replicate trajectories), builds a free-energy landscape from the
# leading two components, clusters the reduced conformational space, extracts
# representative centroid structures, and produces a standardised multi-panel
# report per method. Supports out-of-sample projection of a secondary
# trajectory onto axes fitted from a primary one (e.g. project a mutant or
# apo trajectory onto a WT/holo PCA/TICA/UMAP model).
#
# BACKEND
# -------
# Standalone, MDAnalysis-only. No PyMOL dependency.
#
# RUNNING IT
# ----------
#   Recommended : python quick_pca.py --config my_config.yaml
#   CLI fallback: python quick_pca.py [--help for all options]
#   Supported trajectory formats: .nc  .xtc  .trr  .dcd
#
# OUTPUT (per method: pca / tica / umap)
# ---------------------------------------
#   <prefix>_<METHOD>.png            — combined 4+ panel report
#   <prefix>_<METHOD>_replicates.png — replicate confidence-ellipse plot (multi-traj only)
#   <prefix>_<METHOD>_projected.png  — primary-vs-projected overlay (if projection.groups used)
#   <prefix>_<METHOD>_variance.csv   — explained/kinetic variance + timescales
#   <prefix>_<METHOD>_loadings.csv   — eigenvectors (PCA / TICA only; UMAP has none)
#   <outdir>/clustering/<method>/    — elbow/BIC/silhouette diagnostics, cluster
#                                       scatter plots, per-cluster centroid PDBs
#   <outdir>/individual_panels/      — every report panel saved as its own PNG
#
# DEPENDENCIES
# ------------
#   pip install numpy scikit-learn scipy matplotlib
#   pip install MDAnalysis PyYAML
#   pip install deeptime          # only needed for TICA
#   pip install umap-learn        # only needed for UMAP
#   pip install hdbscan           # only needed for HDBSCAN clustering
#
# NOTE ON ATTRIBUTION
# --------------------
# <<< Replace this block with your own author / license / citation info. >>>
#
# =============================================================================

import argparse
import csv
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
from MDAnalysis.lib.distances import distance_array

# =============================================================================
# ⚙️  DEFAULT SETTINGS  — every one of these is overridden by --config / CLI
#      flags at runtime. They exist so the script also has sane behaviour if
#      run with zero arguments against a folder with one topology/trajectory.
# =============================================================================

# --- Atom selection ----------------------------------------------------------
PCA_SEL_MDA = "protein and name CA"   # MDAnalysis selection string used for the
                                       # reduction itself. CA-only is standard for
                                       # backbone-scale essential dynamics; widen
                                       # this (e.g. "protein and backbone") if you
                                       # need finer detail, at the cost of more
                                       # features per frame.

# --- Dimensionality reduction --------------------------------------------------
METHODS   = ["pca"]       # any subset of: "pca", "tica", "umap"
PCA_MODE  = "count"       # "count"    → keep PCA_NCOMP components (fast, randomized SVD)
                           # "variance" → keep however many components are needed
                           #              to reach PCA_VAR cumulative variance (full SVD)
PCA_NCOMP = 10             # components to compute in "count" mode
PCA_VAR   = 0.90           # cumulative variance threshold in "variance" mode
TICA_LAG  = 10              # TICA lag time, in *strided* frames (i.e. after MD_INTERVAL
                            # is applied) — not in raw simulation frames or time units.

# --- Free-Energy Landscape ----------------------------------------------------
PCA_NBINS = 50             # 2-D histogram bins per axis for the FEL
PCA_SIGMA = 1.0            # Gaussian smoothing sigma, in bins (reduces histogram noise
                            # before the Boltzmann inversion)
PCA_TEMP  = 300.0          # temperature (K) used in the Boltzmann inversion, F = -kT ln(P)
TICA_INPUT_SPACE = "coordinates"  # "coordinates" or "pca"
UMAP_INPUT_SPACE = "pca"           # "coordinates" or "pca"
RANDOM_SEED = 42
UMAP_N_COMPONENTS = 2
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"

# --- Trajectory sampling -------------------------------------------------------
MD_INTERVAL = 1            # use every Nth frame (stride). Increase for very long/dense
                            # trajectories where consecutive frames are highly correlated.

# --- Output ---------------------------------------------------------------------
OUTPUT_PREFIX = "Report"   # method name (PCA/TICA/UMAP) is appended automatically
EXPORT_CSV    = True

# --- Clustering -------------------------------------------------------------------
CLUSTER                    = True
CLUSTER_METHOD             = "kmeans"  # "kmeans", "gmm", or "hdbscan" — see README for
                                        # guidance on which to pick for your data
CLUSTER_USE_UMAP           = False     # additionally re-embed the clustering input with
                                        # UMAP purely for a nicer 2-D cluster visualization
                                        # (does not change the actual cluster assignment
                                        # input unless CLUSTER_USE_UMAP embedding is what's
                                        # clustered — see run_clustering)
CLUSTER_KMAX               = 12        # max k tried for kmeans/gmm model selection
CLUSTER_VARIANCE_THRESHOLD = 0.80      # cluster on however many leading components reach
                                        # this cumulative variance (not necessarily just PC1/PC2)
CLUSTER_INERTIA_THRESHOLD  = 0.15      # kmeans elbow: stop increasing k once the relative
                                        # inertia drop between k and k+1 falls below this
CLUSTER_EXTRACT_CENTROIDS  = True
CLUSTER_OUTDIR             = "clustering"  # resolved under --outdir at runtime
CLUSTER_SILHOUETTE         = False     # compute a (sampled) silhouette score per k — useful
                                        # as a second opinion on k, but O(n^2)-ish, hence sampled
CLUSTER_SILHOUETTE_SAMPLE  = 2000      # max frames used when computing silhouette_score
CLUSTER_SELECT_BY          = "silhouette"  # "elbow" or "silhouette" — which criterion actually
                                            # picks k for kmeans when both are computed
CLUSTER_CENTROID_LIGAND_SELECTION = None   # OPTIONAL MDAnalysis selection for a ligand that
                                            # should be re-imaged next to its nearest protein
                                            # image before a centroid PDB is written (only
                                            # relevant for periodic/membrane systems with a
                                            # ligand that can wrap across the box boundary).
                                            # Leave as None for protein-only or non-periodic
                                            # analyses — the step is skipped entirely.
GMM_COVARIANCE_TYPE        = "full"
GMM_REG_COVAR              = 1e-6
GMM_BIC_DELTA_STOP         = 300        # GMM model selection: stop adding components once
                                         # the BIC improvement from k to k+1 drops below this
GMM_N_INIT                 = 5
HDBSCAN_MIN_CLUSTER_SIZE   = 100
HDBSCAN_MIN_SAMPLES        = None
HDBSCAN_METRIC             = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"
CLUSTER_DIRECTORY_NAME     = "clustering"

# --- Replicate handling (MDAnalysis multi-trajectory) -----------------------------
REPLICATE_PLOT           = True
REPLICATE_ELLIPSE_LEVEL  = 0.95     # confidence level for the replicate covariance ellipses
REPORT_COLUMNS           = 2        # subplot columns in the combined report figure
SAVE_INDIVIDUAL_PANELS   = True     # also save each report panel as its own standalone PNG

# --- Out-of-sample projection (secondary trajectory onto primary model) -----------
PROJECT_LABEL = "Projected"   # default label if a projection group doesn't set its own
PROJECT_PLOT  = True          # save a standalone overlay figure of primary vs. projected

# =============================================================================
# 🎨  PALETTE  (shared across all plots so colors stay consistent)
# =============================================================================

_PALETTE = [
    "steelblue", "coral", "teal", "darkorange",
    "mediumpurple", "seagreen", "crimson", "goldenrod",
    "slategray", "deeppink",
]


# =============================================================================
# 🧪  SECTION 1 — INPUT VALIDATION HELPERS
#      Everything here runs *before* any trajectory frame is touched: making
#      sure the number of topologies/labels/colors line up with the number of
#      trajectories, and that every replicate actually contains the same
#      selected atoms in the same order (otherwise PCA/TICA math is silently
#      wrong — you'd be comparing coordinate vectors for different atoms).
# =============================================================================

def _normalise_optional_list(values, n, name, default_factory):
    """Validate an optional per-trajectory CLI list and supply defaults."""
    if values is None:
        return [default_factory(i) for i in range(n)]
    if len(values) != n:
        raise ValueError(
            f"--{name} requires exactly {n} values (one per trajectory); "
            f"received {len(values)}."
        )
    return list(values)


def _expand_topologies(topologies, trajectories, option_name="--topology"):
    """Return one topology per trajectory, accepting one shared topology or N topologies."""
    if not topologies:
        raise ValueError(f"{option_name} was not provided.")
    if len(topologies) == 1:
        return list(topologies) * len(trajectories)
    if len(topologies) == len(trajectories):
        return list(topologies)
    raise ValueError(
        f"{option_name} must contain either one shared topology or exactly one "
        f"topology per trajectory ({len(trajectories)} expected, {len(topologies)} received)."
    )


def _atom_signature(atomgroup):
    """Topology-independent identity/order signature for a selected atom group."""
    return tuple(
        (int(atom.resid), str(atom.name))
        for atom in atomgroup
    )


def _validate_selected_atom_order(universes, selection, labels=None):
    """
    Ensure every trajectory yields identical selected atoms in identical order.

    This matters because PCA/TICA treat the coordinate vector as a fixed-length
    feature vector — if replicate 2's selection returns atoms in a different
    order (e.g. different chain ordering in the topology file), every frame
    from that replicate would be silently miscompared against replicate 1.
    """
    if not isinstance(universes, (list, tuple)):
        universes = [universes]
    labels = labels or [f"Trajectory {i + 1}" for i in range(len(universes))]

    reference = None
    reference_label = None
    for i, (u, label) in enumerate(zip(universes, labels), start=1):
        ag = u.select_atoms(selection)
        if len(ag) == 0:
            raise ValueError(f"{label}: selection '{selection}' matched 0 atoms.")
        sig = _atom_signature(ag)
        if reference is None:
            reference = sig
            reference_label = label
            continue
        if sig != reference:
            if len(sig) != len(reference):
                detail = f"atom count {len(sig)} vs {len(reference)}"
            else:
                mismatch = next(j for j, pair in enumerate(zip(reference, sig)) if pair[0] != pair[1])
                detail = (
                    f"first mismatch at selected atom {mismatch + 1}: "
                    f"{sig[mismatch]} vs {reference[mismatch]}"
                )
            raise ValueError(
                f"Selected atom identity/order mismatch for {label} relative to "
                f"{reference_label}: {detail}."
            )
    return reference


# =============================================================================
# 🔌  SECTION 2 — ALIGNMENT BACKEND
#      All frames are rigid-body aligned to a reference before reduction, since
#      PCA/TICA on raw lab-frame coordinates would mostly capture translation
#      and rotation rather than internal conformational change.
# =============================================================================

def _kabsch_rotation(mobile, ref_pos, ref_com):
    """Return a NumPy Kabsch rotation matrix for mobile -> reference."""
    H        = (mobile - mobile.mean(0)).T @ (ref_pos - ref_com)
    U, _, Vt = np.linalg.svd(H)
    d        = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


# Prefer MDAnalysis's compiled rotation_matrix (faster than the numpy SVD above);
# only fall back to the pure-numpy Kabsch implementation if that import fails.
try:
    from MDAnalysis.analysis.align import rotation_matrix as _mda_rotmat
    def _alignment_rotation(mobile, ref_pos, ref_com):
        R, _ = _mda_rotmat(mobile - mobile.mean(0), ref_pos - ref_com)
        return R
except ImportError:
    _alignment_rotation = _kabsch_rotation


# =============================================================================
# 🧬  SECTION 3 — FRAME EXTRACTION
#      Walks every trajectory once, aligns each frame, and returns a single
#      (n_frames x n_features) coordinate matrix plus bookkeeping arrays
#      (which replicate / which original frame index each row came from).
#      This is the only place trajectories are actually read frame-by-frame;
#      every downstream method (PCA/TICA/UMAP, clustering, centroid
#      extraction) reuses these arrays.
# =============================================================================

def _extract_frames(
    universes,
    selection,
    align_selection=None,
    ref_pos=None,
    ref_com=None,
    return_reference=False,
    contact_filter_ligand=None,
    contact_filter_protein="protein and not name H*",
    contact_filter_cutoff=4.0,
):
    """
    Extract, align, and (optionally) contact-filter frames from one or more
    trajectories into a single feature matrix.

    align_selection can differ from `selection` (e.g. align on a stable
    subdomain, but reduce on the whole protein) — pass ref_pos/ref_com to reuse
    an existing alignment reference frame, which is how secondary/projected
    trajectories are aligned onto the *same* reference as the primary system.

    contact_filter_ligand, if given, drops frames where the ligand's median
    minimum distance to `contact_filter_protein` exceeds contact_filter_cutoff
    (Å) — e.g. to exclude frames where a ligand has unbound/drifted away
    before running ligand-pocket-focused PCA.
    """
    if not isinstance(universes, (list, tuple)):
        universes = [universes]

    align_selection = align_selection or selection

    n_per_rep = [len(range(0, len(u.trajectory), MD_INTERVAL)) for u in universes]
    n_total = int(sum(n_per_rep))

    ag0 = universes[0].select_atoms(selection)
    align0 = universes[0].select_atoms(align_selection)

    n_at = len(ag0)
    n_align = len(align0)

    if n_at == 0:
        raise ValueError(f"Selection '{selection}' matched 0 atoms.")
    if n_align == 0:
        raise ValueError(f"Align selection '{align_selection}' matched 0 atoms.")

    if ref_pos is not None:
        ref_pos = np.asarray(ref_pos, dtype=np.float64)
        if ref_pos.shape != (n_align, 3):
            raise ValueError(
                "Projection alignment mismatch: secondary align selection has "
                f"{n_align} atoms, but the primary reference has {ref_pos.shape[0]}."
            )
        ref_com = np.asarray(ref_com, dtype=np.float64)

    out = np.empty((n_total, n_at * 3), dtype=np.float64)
    rep_arr = np.empty(n_total, dtype=np.int32)
    frame_arr = np.empty(n_total, dtype=np.int32)
    row = 0

    for rep_idx, u in enumerate(universes, start=1):
        ag = u.select_atoms(selection)
        align_ag = u.select_atoms(align_selection)

        if len(ag) != n_at:
            raise ValueError(
                f"Rep {rep_idx}: PCA selection atom count mismatch ({len(ag)} vs {n_at})."
            )

        if len(align_ag) != n_align:
            raise ValueError(
                f"Rep {rep_idx}: align selection atom count mismatch ({len(align_ag)} vs {n_align})."
            )

        lig_filter = None
        prot_filter = None

        if contact_filter_ligand is not None:
            lig_filter = u.select_atoms(contact_filter_ligand)
            prot_filter = u.select_atoms(contact_filter_protein)

            if len(lig_filter) == 0:
                raise ValueError(
                    f"Contact-filter ligand selection matched 0 atoms: {contact_filter_ligand}"
                )
            if len(prot_filter) == 0:
                raise ValueError(
                    f"Contact-filter protein selection matched 0 atoms: {contact_filter_protein}"
                )

        kept_rep = 0

        for ts_idx in range(0, len(u.trajectory), MD_INTERVAL):
            u.trajectory[ts_idx]

            if lig_filter is not None:
                d = distance_array(
                    lig_filter.positions,
                    prot_filter.positions,
                    box=u.dimensions,
                )

                ligand_atom_min_dists = np.min(d, axis=1)
                median_ligand_distance = np.median(ligand_atom_min_dists)

                if median_ligand_distance > contact_filter_cutoff:
                    continue

            coords = ag.positions.astype(np.float64, copy=False)
            align_coords = align_ag.positions.astype(np.float64, copy=False)

            if ref_pos is None:
                ref_pos = align_coords.copy()
                ref_com = ref_pos.mean(0)

            # Align using align_selection, then apply that transform to selection.
            mobile_com = align_coords.mean(0)
            # Use MDAnalysis's compiled rotation when available and the
            # equivalent NumPy Kabsch implementation otherwise.
            R = _alignment_rotation(align_coords, ref_pos, ref_com)
            aligned_coords = (coords - mobile_com) @ R.T + ref_com

            out[row] = aligned_coords.ravel()
            rep_arr[row] = rep_idx
            frame_arr[row] = ts_idx
            row += 1
            kept_rep += 1

        print(
            f"   Rep {rep_idx} ({os.path.basename(str(u.filename))}): "
            f"kept {kept_rep} / {n_per_rep[rep_idx-1]} frames after stride {MD_INTERVAL}"
        )

    out = out[:row]
    rep_arr = rep_arr[:row]
    frame_arr = frame_arr[:row]

    if row == 0:
        raise ValueError(
            "Contact filter removed all frames. Try a larger cutoff or check selections."
        )

    rep_ids = rep_arr if len(universes) > 1 else None

    if return_reference:
        return out, rep_ids, ref_pos, ref_com, frame_arr
    return out, rep_ids, frame_arr


# =============================================================================
# 🔬  SECTION 4 — DIMENSIONALITY REDUCTION (PCA / TICA / UMAP)
#      Each reduce_* function takes the extracted coordinate matrix and
#      returns a standardised result dict so downstream plotting/clustering
#      code doesn't need to know which method produced it. Common keys:
#        projections               — (n_frames, n_components) reduced coords
#        n_components              — number of components actually kept
#        comp_label                — axis label prefix ("PC" / "IC" / "Dim")
#        explained_variance_ratio  — per-component variance fraction (None for UMAP)
#        eigenvectors              — loadings, only for PCA/TICA (linear methods)
#        model                     — fitted sklearn/deeptime/umap model, kept so
#                                     secondary trajectories can be projected later
# =============================================================================

def _cross_corr(evecs, weights, n_atoms):
    """Residue cross-correlation matrix from eigenvectors + eigenvalue weights."""
    evecs_3d   = evecs.reshape(len(evecs), n_atoms, 3)
    cov        = np.einsum('kia,kja,k->ij', evecs_3d, evecs_3d, np.abs(weights))
    var        = np.diag(cov)
    denom      = np.sqrt(np.outer(var, var))
    return np.where(denom > 0, cov / denom, 0.0).astype(np.float32)


def reduce_pca(positions, n_components=None):
    """
    PCA on the (n_frames x 3N) Cα coordinate matrix.

    Uses randomized SVD in "count" mode (15-17x faster for a fixed small
    n_components on large frame counts) and falls back to full SVD in
    "variance" mode, since randomized SVD doesn't reliably expose enough
    components up front to know where the variance threshold is crossed.
    """
    from sklearn.decomposition import PCA as _PCA

    n_components = PCA_NCOMP if n_components is None else n_components
    if PCA_MODE == "variance":
        n_comp, solver = PCA_VAR, "full"
    else:
        n_comp = min(n_components, min(positions.shape) - 1)
        solver = "randomized"

    centered = positions - positions.mean(0)
    model    = _PCA(n_components=n_comp, svd_solver=solver, random_state=RANDOM_SEED)
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


def reduce_tica(frame_arrays, lag=None, n_components=None,
                cartesian_components=None, n_cartesian_atoms=None):
    """
    TICA via deeptime. *frame_arrays* is a list of (n_frames, n_features) arrays
    — one per trajectory — so TICA respects trajectory boundaries (it must not
    treat the last frame of replicate 1 and the first frame of replicate 2 as
    a continuous kinetic transition).
    """
    try:
        from deeptime.decomposition import TICA as _TICA
    except ImportError:
        print("❌  deeptime not found.  pip install deeptime")
        return None

    lag = TICA_LAG if lag is None else lag
    n_components = PCA_NCOMP if n_components is None else n_components
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
    if cartesian_components is not None:
        # Compose TICA loadings with the upstream PCA loadings so exported
        # eigenvectors and atom cross-correlation remain in Cartesian space.
        evecs = evecs @ cartesian_components
        n_cc_atoms = n_cartesian_atoms
    else:
        n_cc_atoms = n_feat // 3
    cc     = _cross_corr(evecs, sv2, n_cc_atoms)
    ts     = model.timescales(lagtime=lag)[:n_comp]

    print(f"   IC1={evr[0]*100:.1f}%  IC2={evr[1]*100:.1f}%  "
          f"top-{n_comp} cumul={cumvar[-1]*100:.1f}%")
    print(f"   Implied timescales (frames): {np.round(ts[:5], 1)}")

    return dict(projections=proj, explained_variance_ratio=evr,
                cumulative_variance=cumvar, cross_correlation=cc,
                n_components=n_comp, eigenvectors=evecs,
                model=model, timescales=ts, lag=lag,
                bar_label="Kinetic Variance (%)", comp_label="IC")


def reduce_umap(projections, n_components=None):
    """UMAP applied to existing projections (PCA or TICA output). No eigenvectors —
    UMAP is a non-linear embedding, so there's no per-residue loading to report."""
    try:
        import umap as _umap
    except ImportError:
        print("❌  umap-learn not found.  pip install umap-learn")
        return None
    print("   Running UMAP …")
    n_components = n_components or UMAP_N_COMPONENTS
    model = _umap.UMAP(
        n_components=n_components,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_SEED,
    )
    emb = model.fit_transform(projections)
    return dict(projections=emb, n_components=n_components,
                model=model, upstream_projections=projections,
                bar_label=None, comp_label="Dim")


def _reduce(method, positions, frame_arrays):
    """Dispatch to the correct reduction function and return a standardised result dict."""
    if method == "pca":
        return reduce_pca(positions)
    if method == "tica":
        if TICA_INPUT_SPACE == "pca":
            pca_r = reduce_pca(positions)
            pca_proj = pca_r["projections"]
            offset = 0
            pca_frame_arrays = []
            for frames in frame_arrays:
                pca_frame_arrays.append(pca_proj[offset:offset + len(frames)])
                offset += len(frames)
            r = reduce_tica(
                pca_frame_arrays,
                cartesian_components=pca_r["eigenvectors"],
                n_cartesian_atoms=positions.shape[1] // 3,
            )
            if r is not None:
                r["upstream_model"] = pca_r["model"]
                r["upstream_fit_mean"] = pca_r["fit_mean"]
                r["input_space"] = "pca"
            return r
        r = reduce_tica(frame_arrays)
        if r is not None:
            r["input_space"] = "coordinates"
        return r
    if method == "umap":
        if UMAP_INPUT_SPACE == "pca":
            pca_r = reduce_pca(positions)
            r = reduce_umap(pca_r["projections"]) if pca_r else None
            if r is not None:
                r["upstream_model"] = pca_r.get("model")
                r["upstream_fit_mean"] = pca_r.get("fit_mean")
                r["input_space"] = "pca"
        else:
            r = reduce_umap(positions)
            if r is not None:
                r["input_space"] = "coordinates"
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
        if result.get("input_space") == "pca":
            upstream = result.get("upstream_model")
            upstream_mean = result.get("upstream_fit_mean")
            if upstream is None or upstream_mean is None:
                raise ValueError("PCA-preprocessed TICA projection requires the upstream PCA model.")
            positions = upstream.transform(positions - upstream_mean)
        return model.transform(positions)
    if method == "umap":
        if not hasattr(model, "transform"):
            raise ValueError("The installed umap-learn model does not support transform().")
        if result.get("input_space") == "pca":
            upstream = result.get("upstream_model")
            upstream_mean = result.get("upstream_fit_mean")
            if upstream is None or upstream_mean is None:
                raise ValueError("PCA-preprocessed UMAP projection requires the upstream PCA model.")
            positions = upstream.transform(positions - upstream_mean)
        return model.transform(positions)

    raise ValueError(f"Unknown projection method: {method}")


def _attach_projection(result, projected_positions, projected_rep_ids, method, label):
    """Append one out-of-sample projection group to a method result dict."""
    proj = _project_positions(method, result, projected_positions)

    if "projected" not in result or result["projected"] is None:
        result["projected"] = []

    result["projected"].append({
        "label": label,
        "projections": np.asarray(proj, dtype=np.float64),
        "replicate_ids": projected_rep_ids,
    })

    return result


# =============================================================================
# 🌊  SECTION 5 — FREE-ENERGY LANDSCAPE
# =============================================================================

def compute_fel(result, temperature=None, n_bins=None, sigma=None):
    """Boltzmann inversion of PC1/PC2 (or IC1/IC2, Dim1/Dim2) density: F = -kT ln(P)."""
    temperature = PCA_TEMP if temperature is None else temperature
    n_bins = PCA_NBINS if n_bins is None else n_bins
    sigma = PCA_SIGMA if sigma is None else sigma
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
# 🔵  SECTION 6 — CLUSTERING
#      Three interchangeable cluster backends (kmeans / gmm / hdbscan), model
#      selection diagnostics for each, and centroid PDB extraction.
# =============================================================================

def _kmeans_elbow(projections, method_label):
    """
    KMeans elbow, optional quick sampled silhouette, saves diagnostics,
    returns (labels, centres).
    """
    from sklearn.cluster import KMeans

    ks = list(range(2, CLUSTER_KMAX + 1))
    fitted = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init="auto").fit(projections)
        fitted.append(km)

    inertias = np.array([km.inertia_ for km in fitted])
    drops = -np.diff(inertias) / inertias[:-1]

    best_k = ks[0]
    for i, d in enumerate(drops):
        if d >= CLUSTER_INERTIA_THRESHOLD:
            best_k = ks[i + 1]
        else:
            break

    silhouette_scores = None
    if CLUSTER_SILHOUETTE:
        from sklearn.metrics import silhouette_score

        n = len(projections)
        sample_size = min(CLUSTER_SILHOUETTE_SAMPLE, n)
        silhouette_scores = []

        for k, km in zip(ks, fitted):
            # silhouette_score is O(n^2) without sampling, so keep this bounded.
            score = silhouette_score(
                projections,
                km.labels_,
                metric="euclidean",
                sample_size=sample_size if sample_size < n else None,
                random_state=RANDOM_SEED,
            )
            silhouette_scores.append(score)

        silhouette_scores = np.asarray(silhouette_scores, dtype=float)
        best_sil_idx = int(np.nanargmax(silhouette_scores))
        print(
            f"   🧪  Silhouette best k={ks[best_sil_idx]} "
            f"(score={silhouette_scores[best_sil_idx]:.3f}, "
            f"sample={sample_size}/{n})"
        )

        sil_csv = os.path.join(CLUSTER_OUTDIR, f"silhouette_{method_label}.csv")
        np.savetxt(
            sil_csv,
            np.column_stack([ks, inertias, silhouette_scores]),
            delimiter=",",
            header="k,inertia,silhouette_score",
            comments="",
            fmt=["%d", "%.6f", "%.6f"],
        )
        print(f"   🗂️  Silhouette scores → {sil_csv}")

    print(f"   🔵  {method_label} elbow best k={best_k}  "
          f"(threshold {CLUSTER_INERTIA_THRESHOLD:.0%})")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, inertias, "o-", color="steelblue", lw=2, ms=6, mec="navy", mew=0.8)
    for i in range(len(inertias) - 1):
        ax.text((ks[i]+ks[i+1])/2+0.1, (inertias[i]+inertias[i+1])/2,
                f"{drops[i]:.1%}", fontsize=8, ha="center", va="bottom")
    ax.axvline(best_k, color="coral", ls="--", lw=1.4, label=f"Elbow k={best_k}")

    if silhouette_scores is not None:
        ax2 = ax.twinx()
        ax2.plot(ks, silhouette_scores, "s--", color="teal", lw=1.6, ms=5,
                 label="Silhouette")
        ax2.set_ylabel("Silhouette score", fontsize=10, color="teal")
        ax2.tick_params(axis="y", labelcolor="teal")
        best_sil_k = ks[int(np.nanargmax(silhouette_scores))]
        ax2.axvline(best_sil_k, color="teal", ls=":", lw=1.2,
                    label=f"Silhouette k={best_sil_k}")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, fontsize=9)
    else:
        ax.legend(fontsize=9)

    ax.set_xlabel("k", fontsize=11, fontweight="bold")
    ax.set_ylabel("Inertia", fontsize=11, fontweight="bold")
    ax.set_title(f"Elbow — {method_label}", fontsize=12, fontweight="bold")
    os.makedirs(CLUSTER_OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(CLUSTER_OUTDIR, f"elbow_{method_label}.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    if CLUSTER_SILHOUETTE and CLUSTER_SELECT_BY == "silhouette":
        valid = np.isfinite(silhouette_scores)

        if np.any(valid):
            best_k = int(np.array(ks)[valid][np.argmax(silhouette_scores[valid])])
            print(f"   🔵  {method_label} best k={best_k} by silhouette")
        else:
            print("   ⚠️  No valid silhouette scores — using elbow k")
    else:
        print(f"   🔵  {method_label} best k={best_k} by elbow")

    best_i = ks.index(best_k)
    return fitted[best_i].labels_, fitted[best_i].cluster_centers_


def _gmm_bic(projections, method_label):
    """
    Fit GMMs for k=2..CLUSTER_KMAX.

    Select the first k after which adding one more component improves BIC
    by less than GMM_BIC_DELTA_STOP. If that never happens, use the global
    minimum-BIC model.
    """
    from sklearn.mixture import GaussianMixture

    ks = list(range(2, CLUSTER_KMAX + 1))
    fitted = []
    bics = []
    aics = []

    for k in ks:
        model = GaussianMixture(
            n_components=k,
            covariance_type=GMM_COVARIANCE_TYPE,
            reg_covar=GMM_REG_COVAR,
            n_init=GMM_N_INIT,
            random_state=RANDOM_SEED,
        ).fit(projections)

        fitted.append(model)
        bics.append(model.bic(projections))
        aics.append(model.aic(projections))

    bics = np.asarray(bics, dtype=float)
    aics = np.asarray(aics, dtype=float)

    # Positive value means the larger model improved BIC.
    # Example:
    # BIC(k=2)=1000, BIC(k=3)=900 -> improvement = 100
    bic_improvements = bics[:-1] - bics[1:]

    # Default fallback: global minimum BIC.
    best_idx = int(np.argmin(bics))
    selection_reason = "minimum BIC"

    # Stop at the simpler model when the next component gives only a small gain.
    for i, improvement in enumerate(bic_improvements):
        if improvement < GMM_BIC_DELTA_STOP:
            best_idx = i
            selection_reason = (
                f"ΔBIC early stop: improvement from k={ks[i]} "
                f"to k={ks[i + 1]} was {improvement:.2f}, "
                f"below threshold {GMM_BIC_DELTA_STOP:.2f}"
            )
            break

    best_k = ks[best_idx]
    best_model = fitted[best_idx]

    os.makedirs(CLUSTER_OUTDIR, exist_ok=True)

    # Add ΔBIC to the output table. The final row has no next model.
    delta_column = np.full(len(ks), np.nan, dtype=float)
    delta_column[:-1] = bic_improvements

    csv_path = os.path.join(
        CLUSTER_OUTDIR,
        f"gmm_model_selection_{method_label}.csv",
    )

    np.savetxt(
        csv_path,
        np.column_stack([ks, bics, aics, delta_column]),
        delimiter=",",
        header="k,bic,aic,bic_improvement_to_next_k",
        comments="",
        fmt=["%d", "%.6f", "%.6f", "%.6f"],
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(ks, bics, "o-", label="BIC")
    ax.plot(ks, aics, "s--", label="AIC")
    ax.axvline(
        best_k,
        linestyle=":",
        label=f"Selected k={best_k}",
    )

    ax.set_xlabel("Number of components")
    ax.set_ylabel("Information criterion")
    ax.set_title(f"GMM Model Selection — {method_label}")
    ax.legend()

    fig.savefig(
        os.path.join(
            CLUSTER_OUTDIR,
            f"gmm_model_selection_{method_label}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"   🔵  {method_label} GMM selected k={best_k}")
    print(f"       Selection rule: {selection_reason}")
    print(f"   🗂️  GMM model selection → {csv_path}")

    return best_model.predict(projections), best_model.means_


def _hdbscan_cluster(projections, method_label):
    """Density-based clustering; noise/transition frames receive label -1."""
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError(
            "HDBSCAN clustering requires the 'hdbscan' package. "
            "Install it with: pip install hdbscan"
        ) from exc

    model = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
        prediction_data=True,
    )
    raw_labels = model.fit_predict(projections)
    cluster_ids = [int(x) for x in sorted(np.unique(raw_labels)) if x != -1]
    if not cluster_ids:
        raise ValueError(
            "HDBSCAN found no clusters. Reduce min_cluster_size or min_samples."
        )

    label_map = {old: new for new, old in enumerate(cluster_ids)}
    labels = np.full(len(raw_labels), -1, dtype=int)
    for old, new in label_map.items():
        labels[raw_labels == old] = new

    centers = []
    for cluster_id in range(len(cluster_ids)):
        idx = np.where(labels == cluster_id)[0]
        points = projections[idx]
        mean_position = points.mean(axis=0)
        centers.append(points[np.argmin(np.linalg.norm(points - mean_position, axis=1))])
    centers = np.asarray(centers)

    os.makedirs(CLUSTER_OUTDIR, exist_ok=True)
    csv_path = os.path.join(CLUSTER_OUTDIR, f"hdbscan_membership_{method_label}.csv")
    np.savetxt(
        csv_path,
        np.column_stack([np.arange(len(labels)), labels, model.probabilities_, model.outlier_scores_]),
        delimiter=",", header="frame,cluster,membership_probability,outlier_score",
        comments="", fmt=["%d", "%d", "%.6f", "%.6f"],
    )
    n_noise = int(np.sum(labels == -1))
    print(
        f"   🔵  {method_label} HDBSCAN found {len(centers)} clusters; "
        f"{n_noise}/{len(labels)} frames marked as noise"
    )
    print(f"   🗂️  HDBSCAN membership → {csv_path}")
    return labels, centers


def _select_clustering_dimensions(result, projections, method_label):
    """Select the smallest leading component set reaching the variance threshold.

    PCA uses explained variance and TICA uses kinetic variance. Embeddings without
    variance fractions (for example a final UMAP result) retain all dimensions.
    At least two dimensions are retained when available so existing 2-D cluster
    diagnostics remain valid.
    """
    projections = np.asarray(projections)
    if projections.ndim != 2:
        raise ValueError("Clustering projections must be a 2-D array.")

    evr = result.get("explained_variance_ratio")
    if evr is None:
        print(
            f"   ℹ️  {method_label}: no component variance fractions are available; "
            f"using all {projections.shape[1]} dimensions for clustering."
        )
        return projections, projections.shape[1], None

    threshold = float(CLUSTER_VARIANCE_THRESHOLD)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            "clustering.variance_threshold must be greater than 0 and no greater than 1."
        )

    evr = np.asarray(evr, dtype=float)
    n_available = min(projections.shape[1], len(evr))
    if n_available == 0:
        raise ValueError("No components are available for clustering.")

    cumulative = np.cumsum(evr[:n_available])
    n_selected = int(np.searchsorted(cumulative, threshold, side="left") + 1)
    n_selected = min(n_selected, n_available)

    # Preserve the existing two-dimensional diagnostics whenever possible.
    if n_available >= 2 and n_selected < 2:
        n_selected = 2

    captured = float(cumulative[n_selected - 1])
    variance_name = "kinetic variance" if result.get("comp_label") == "IC" else "variance"
    print(
        f"   📐  {method_label}: clustering on the first {n_selected} component(s), "
        f"capturing {captured:.1%} cumulative {variance_name} "
        f"(target {threshold:.1%})."
    )
    return projections[:, :n_selected], n_selected, captured


def _cluster_embedding(projections, method_label):
    """Dispatch clustering while preserving the labels/centers workflow."""
    method = CLUSTER_METHOD.lower()
    if method == "kmeans":
        return _kmeans_elbow(projections, method_label)
    if method == "gmm":
        return _gmm_bic(projections, method_label)
    if method == "hdbscan":
        return _hdbscan_cluster(projections, method_label)
    raise ValueError(
        f"Unknown clustering method '{CLUSTER_METHOD}'. Choose kmeans, gmm, or hdbscan."
    )


def _cluster_color(cluster_id):
    """Return a stable color; HDBSCAN noise is displayed in light gray."""
    if int(cluster_id) == -1:
        return "lightgray"
    return _PALETTE[int(cluster_id) % len(_PALETTE)]


def _plot_cluster_embedding(emb2d, labels, centers2d, title, fname):
    """2-D scatter coloured by cluster. Matches report aesthetics."""
    unique_k = np.unique(labels)
    fig, ax  = plt.subplots(figsize=(8, 7))
    ax.scatter(emb2d[:, 0], emb2d[:, 1],
               c=[_cluster_color(i) for i in labels],
               edgecolors="k", linewidths=0.3, s=40, alpha=0.85, rasterized=True)
    for i, ctr in enumerate(centers2d):
        base  = np.array(mcolors.to_rgb(_PALETTE[i % len(_PALETTE)]))
        light = base + (1 - base) * 0.45
        ax.scatter(ctr[0], ctr[1], color=light, edgecolors="k",
                   s=220, marker="X", lw=1.5, zorder=10)
        ax.text(ctr[0], ctr[1], str(i), fontsize=9, fontweight="bold",
                ha="center", va="center", color="white", zorder=11)
    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            label="Noise" if i == -1 else f"Cluster {i}",
            markerfacecolor=_cluster_color(i), markeredgecolor="k", ms=8,
        )
        for i in unique_k
    ]
    ax.legend(handles=legend_handles, title="Clusters", fontsize=9, title_fontsize=10)
    ax.set_xlabel("Dim 1", fontsize=11, fontweight="bold")
    ax.set_ylabel("Dim 2", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    os.makedirs(CLUSTER_OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(CLUSTER_OUTDIR, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊  {fname}")


def _image_ligand_near_protein(u, ligand_sel, protein_sel="protein"):
    """
    Shift ligand by whole box vectors so it is closest to the nearest protein image.
    Only used cosmetically, for saved centroid PDB visualization in periodic systems
    where a ligand can wrap across the box boundary. If `ligand_sel` is None, or
    matches no atoms (e.g. a protein-only system), this is a no-op.
    """
    if not ligand_sel:
        return

    lig = u.select_atoms(ligand_sel)
    prot = u.select_atoms(protein_sel)

    if len(lig) == 0 or len(prot) == 0:
        return

    box = u.dimensions[:3]
    if np.any(box <= 0):
        return

    lig_com = lig.center_of_geometry()

    # Find nearest protein atom under minimum-image convention
    delta = lig_com[None, :] - prot.positions
    delta_mic = delta - box * np.round(delta / box)

    nearest_i = np.argmin(np.sum(delta_mic**2, axis=1))
    nearest_prot_pos = prot.positions[nearest_i]

    # Move ligand COM near that nearest protein atom
    raw_delta = lig_com - nearest_prot_pos
    shift = -box * np.round(raw_delta / box)

    lig.positions += shift


def _extract_centroids(systems, embedding, labels, centers, rep_ids, frame_indices,
                        label, ligand_selection=None):
    """Save centroid PDB for each cluster. Handles single and multi-traj."""
    if not isinstance(systems, (list, tuple)):
        systems = [systems]

    if frame_indices is None:
        raise ValueError("frame_indices missing; cannot extract correct filtered centroid frames.")

    outdir = os.path.join(CLUSTER_OUTDIR, f"centroids_{label}")
    os.makedirs(outdir, exist_ok=True)

    for cid, center in enumerate(centers):
        idx = np.where(labels == cid)[0]
        frame_idx = int(idx[np.argmin(np.linalg.norm(embedding[idx] - center, axis=1))])
        original_ts_idx = int(frame_indices[frame_idx])

        out_path = os.path.join(outdir, f"cluster{cid}.pdb")

        if rep_ids is not None:
            rep = int(rep_ids[frame_idx])
            u = systems[rep - 1]
        else:
            rep = 1
            u = systems[0]

        u.trajectory[original_ts_idx]

        # Cosmetic-only step for centroid PDBs; skipped unless the caller
        # explicitly configured a ligand selection (see
        # CLUSTER_CENTROID_LIGAND_SELECTION / clustering.centroid_ligand_selection).
        _image_ligand_near_protein(
            u,
            ligand_sel=ligand_selection,
            protein_sel="protein",
        )

        u.atoms.write(out_path)

        print(
            f"   💾  Cluster {cid} centroid "
            f"(PCA row {frame_idx}, rep {rep}, traj frame {original_ts_idx}) → {out_path}"
        )


def run_clustering(result, systems, method_label):
    """
    Configurable clustering (+ optional UMAP re-embedding) on result projections.
    Supports KMeans, full-covariance GMM, and HDBSCAN.
    """
    if not CLUSTER:
        return
    proj    = result["projections"]
    rep_ids = result.get("replicate_ids")
    os.makedirs(CLUSTER_OUTDIR, exist_ok=True)
    print(f"\n🔵  Clustering ({method_label}) …")

    clustering_results = {}
    cluster_input, n_cluster_dims, captured_variance = _select_clustering_dimensions(
        result, proj, method_label
    )

    embeddings = {method_label: cluster_input}
    if CLUSTER_USE_UMAP:
        try:
            import umap as _umap
            embeddings[f"{method_label}_UMAP"] = (
                _umap.UMAP(
                    n_components=UMAP_N_COMPONENTS,
                    n_neighbors=UMAP_N_NEIGHBORS,
                    min_dist=UMAP_MIN_DIST,
                    metric=UMAP_METRIC,
                    random_state=RANDOM_SEED,
                ).fit_transform(cluster_input))
        except ImportError:
            print("   ⚠️  umap-learn not found — skipping UMAP.  pip install umap-learn")

    for emb_label, emb in embeddings.items():
        labels, centers = _cluster_embedding(emb, emb_label)
        _plot_cluster_embedding(emb[:, :2], labels, centers[:, :2],
                                title=f"Clusters — {emb_label}",
                                fname=f"clusters_{emb_label}.png")
        csv_path = os.path.join(CLUSTER_OUTDIR, f"labels_{emb_label}.csv")
        frame_indices = result.get("frame_indices")
        if frame_indices is None:
            raise ValueError("frame_indices missing; cannot export traceable cluster labels.")
        metadata = result.get("trajectory_metadata") or []
        export_rep_ids = rep_ids if rep_ids is not None else np.ones(len(labels), dtype=int)
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "dataset_row", "replicate", "original_frame", "trajectory_label",
                "topology", "trajectory", "cluster",
            ])
            for row_idx, (rep, original_frame, cluster_label) in enumerate(
                zip(export_rep_ids, frame_indices, labels)
            ):
                rep = int(rep)
                item = metadata[rep - 1] if rep - 1 < len(metadata) else {}
                writer.writerow([
                    row_idx,
                    rep,
                    int(original_frame),
                    item.get("label", f"Trajectory {rep}"),
                    item.get("topology", ""),
                    item.get("trajectory", ""),
                    int(cluster_label),
                ])
        print(f"   🗂️  Labels → {csv_path}")
        clustering_results[emb_label] = {
            "embedding": emb,
            "labels": labels,
            "centers": centers,
            "input_dimensions": n_cluster_dims,
            "captured_variance": captured_variance,
            "variance_threshold": CLUSTER_VARIANCE_THRESHOLD,
        }

        if CLUSTER_EXTRACT_CENTROIDS:
            _extract_centroids(
                systems,
                emb,
                labels,
                centers,
                rep_ids,
                frame_indices,
                emb_label,
                ligand_selection=CLUSTER_CENTROID_LIGAND_SELECTION,
            )
    return clustering_results


# =============================================================================
# 📊  SECTION 7 — REPORT PANELS
#      One function per panel type. plot_report() (Section 8) decides which
#      panels apply to a given method/result and assembles them into a grid.
# =============================================================================

def _panel_fel(ax, fig, fel, result, method):
    """Panel: Free-Energy Landscape (identical for all methods)."""
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
    Panel:
      PCA / TICA -> residue cross-correlation matrix
      UMAP       -> trajectory-progression scatter (time-coloured)
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
    Panel:
      PCA  -> explained-variance bar chart
      TICA -> implied timescales bar chart
      UMAP -> not shown (non-linear, no EVR to report)
    """
    nc      = result["n_components"]
    n_show  = min(nc, 10)
    bar_lbl = result.get("bar_label")
    evr     = result.get("explained_variance_ratio")

    if "timescales" in result:
        # TICA implied timescales - clip negatives (unphysical; arise from
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
    """Panel: 1-D projection histograms + KDE (identical for all methods)."""
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
    PCA -> PC1/PC2, TICA -> IC1/IC2. Not shown for UMAP (no eigenvectors).
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
        c=[_cluster_color(i) for i in labels],
        edgecolors="k", linewidths=0.25, s=22, alpha=0.75,
        rasterized=True
    )

    for i, ctr in enumerate(centers):
        ax.scatter(
            ctr[0], ctr[1],
            color=_cluster_color(i),
            edgecolors="k", s=160, marker="X", lw=1.2, zorder=10
        )
        ax.text(ctr[0], ctr[1], str(i), fontsize=8, fontweight="bold",
                ha="center", va="center", color="white", zorder=11)

    ax.set_xlabel("Dim 1", fontsize=11, fontweight="bold")
    ax.set_ylabel("Dim 2", fontsize=11, fontweight="bold")
    ax.set_title(f"Clusters — {key}", fontsize=12, fontweight="bold")

    return True


def _axis_labels(result):
    """Consistent axis labels for the first two components/dimensions."""
    comp_lbl = result.get("comp_label", "Dim")
    evr = result.get("explained_variance_ratio")
    xlabel = f"{comp_lbl}1 ({evr[0]*100:.1f}%)" if evr is not None else f"{comp_lbl}1"
    ylabel = f"{comp_lbl}2 ({evr[1]*100:.1f}%)" if evr is not None else f"{comp_lbl}2"
    return xlabel, ylabel


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


def _panel_replicates(ax, result, method):
    """Replicate scatter and confidence ellipses using user metadata when available."""
    rep_ids = result.get("replicate_ids")
    if rep_ids is None:
        return False

    proj = result["projections"]
    metadata = result.get("trajectory_metadata")
    unique_ids = np.unique(rep_ids)
    if metadata is not None and len(metadata) != len(unique_ids):
        raise ValueError(
            "trajectory_metadata length does not match the number of trajectories."
        )

    for i, rep in enumerate(unique_ids):
        mask = rep_ids == rep
        if metadata is None:
            color = _PALETTE[i % len(_PALETTE)]
            label = f"Rep {rep}"
        else:
            color = metadata[i].get("color") or _PALETTE[i % len(_PALETTE)]
            label = metadata[i].get("label") or f"Rep {rep}"

        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            s=20, alpha=0.65, color=color,
            edgecolors="k", lw=0.25, rasterized=True,
            label=f"{label}  (n={mask.sum()})",
        )
        _confidence_ellipse(
            ax, proj[mask, 0], proj[mask, 1],
            color=color, level=REPLICATE_ELLIPSE_LEVEL,
        )

    xlabel, ylabel = _axis_labels(result)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(f"{method.upper()} — Replicate Ellipses",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    ax.set_axisbelow(True)
    return True


def _panel_projection_overlay_one(ax, result, method, projected):
    """One primary-vs-secondary overlay scatter (used both standalone and in-report)."""
    primary = result["projections"]
    secondary = projected["projections"]
    label = projected.get("label", "Projected")

    ax.scatter(
        primary[:, 0], primary[:, 1],
        s=18, alpha=0.30, color="slategray",
        edgecolors="none", rasterized=True,
        label=f"Fit system (n={len(primary)})"
    )

    ax.scatter(
        secondary[:, 0], secondary[:, 1],
        s=24, alpha=0.78, color="crimson",
        edgecolors="k", linewidths=0.25,
        rasterized=True,
        label=f"{label} (n={len(secondary)})"
    )

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
    ax.set_title(f"{label} projected onto {method.upper()} axes",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=True)
    return True


# =============================================================================
# 📄  SECTION 8 — REPORT ASSEMBLY & STANDALONE FIGURES
# =============================================================================

def plot_projection_overlay(result, method, output):
    """Standalone primary-vs-projected overlay plot (one panel per projection group)."""
    projected_items = result.get("projected")
    if not projected_items:
        return

    if isinstance(projected_items, dict):
        projected_items = [projected_items]

    n = len(projected_items)
    fig, axes = plt.subplots(
        1, n,
        figsize=(8 * n, 7),
        squeeze=False,
    )

    for ax, projected in zip(axes.ravel(), projected_items):
        _panel_projection_overlay_one(ax, result, method, projected)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊  Projection overlay → {output}")


def _safe_panel_name(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")


def _save_individual_panel(panel_name, panel_fn, output_dir, method):
    fig, ax = plt.subplots(figsize=(8, 7))
    shown = panel_fn(ax, fig)
    if shown is False:
        plt.close(fig)
        return
    fig.tight_layout()
    path = os.path.join(
        output_dir,
        f"{method.upper()}_{_safe_panel_name(panel_name)}.png",
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊  Individual panel → {path}")


def plot_report(result, method, output, ncols=REPORT_COLUMNS,
                save_individual=SAVE_INDIVIDUAL_PANELS):
    """
    Create the combined multi-panel report and optionally save every panel
    separately. Panels are added conditionally based on what's present in
    `result` — e.g. the replicate-ellipse panel only appears for multi-
    trajectory runs, the clusters panel only if clustering ran, etc. — so a
    single-trajectory PCA run and a 5-replicate TICA-with-projection run each
    get exactly the panels that apply to them.
    """
    fel = compute_fel(result)
    panels = [("free_energy_landscape",
               lambda ax, fig: _panel_fel(ax, fig, fel, result, method))]

    projected_items = result.get("projected") or []
    if isinstance(projected_items, dict):
        projected_items = [projected_items]
    for i, projected in enumerate(projected_items, start=1):
        label = projected.get("label", f"projected_{i}")
        panels.append((
            f"projection_{label}",
            lambda ax, fig, projected=projected:
                _panel_projection_overlay_one(ax, result, method, projected),
        ))

    if result.get("replicate_ids") is not None:
        panels.append(("replicate_ellipses",
                       lambda ax, fig: _panel_replicates(ax, result, method)))
    if result.get("clustering"):
        panels.append(("clusters",
                       lambda ax, fig: _panel_clusters(ax, result, method)))
    if "eigenvectors" in result:
        panels.append(("top_loadings",
                       lambda ax, fig: _panel_top_loadings(ax, result)))
    if method == "pca" and "cross_correlation" in result:
        panels.append(("cross_correlation",
                       lambda ax, fig: _panel_corr_or_time(ax, fig, result, method)))
    if "timescales" in result or result.get("explained_variance_ratio") is not None:
        panels.append(("variance_or_timescales",
                       lambda ax, fig: _panel_variance_or_timescales(ax, result)))
    panels.append(("projection_distributions",
                   lambda ax, fig: _panel_kde(ax, result)))

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(panels) / ncols))
    fig = plt.figure(figsize=(8 * ncols, 6.5 * nrows))
    fig.suptitle(f"Essential Dynamics — {method.upper()} Report",
                 fontsize=15, fontweight="bold")
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.32, wspace=0.35,
                  top=0.94, bottom=0.06, left=0.07, right=0.97)

    for i, (_, panel_fn) in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        panel_fn(ax, fig)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"👑  {method.upper()} report saved → {output}")

    if save_individual:
        panel_dir = os.path.join(os.path.dirname(output) or ".", "individual_panels")
        os.makedirs(panel_dir, exist_ok=True)
        for panel_name, panel_fn in panels:
            _save_individual_panel(panel_name, panel_fn, panel_dir, method)

    _open_file(output)


def plot_replicate_embedding(result, method, output):
    """Save the same replicate renderer used inside the combined report, standalone."""
    if result.get("replicate_ids") is None:
        print("ℹ️  No replicate metadata — skipping replicate plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    _panel_replicates(ax, result, method)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊  Replicate plot → {output}")


# =============================================================================
# 🗂️  SECTION 9 — CSV EXPORT
# =============================================================================

def export_csv(result, method, prefix):
    """Save variance/timescale and loadings CSVs for the given method."""
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
        projected_items = result["projected"]
        if isinstance(projected_items, dict):
            projected_items = [projected_items]
        for group_index, projected in enumerate(projected_items, start=1):
            raw = projected["projections"]
            proj = raw[:, :min(2, raw.shape[1])]
            rep_ids = projected.get("replicate_ids")
            if rep_ids is None:
                rep_ids = np.ones(len(proj), dtype=np.int32)
            rows = np.column_stack([np.arange(len(proj)), rep_ids, proj])
            header = "frame,project_replicate," + ",".join(
                f"{result.get('comp_label', 'Dim')}{i+1}" for i in range(proj.shape[1])
            )
            safe_label = "".join(
                c if c.isalnum() or c in "-_" else "_"
                for c in projected.get("label", f"group{group_index}")
            ).strip("_") or f"group{group_index}"
            np.savetxt(
                f"{prefix}_{method}_projected_{safe_label}.csv",
                rows, delimiter=",", header=header, comments="",
                fmt=["%d", "%d"] + ["%.6f"] * proj.shape[1],
            )

    print(f"   🗂️  CSV → {prefix}_{method}_variance.csv  "
          f"(+ _loadings.csv)" if "eigenvectors" in result else "")


# =============================================================================
# 🚀  SECTION 10 — MAIN PIPELINE
# =============================================================================

def _open_file(path):
    """Open output file in the default viewer only when a display is available."""
    time.sleep(0.3)

    if platform.system() == "Darwin":
        os.system(f"open {path}")
    elif platform.system() == "Windows":
        os.system(f"start {path}")
    elif os.environ.get("DISPLAY"):
        os.system(f"xdg-open {path}")


def _run_pipeline(
    systems,
    selection,
    methods,
    prefix,
    outdir=".",
    projected_systems=None,
    projected_label=PROJECT_LABEL,
    contact_filter_ligand=None,
    contact_filter_protein="protein and not name H*",
    contact_filter_cutoff=4.0,
    align_selection=None,
    trajectory_metadata=None,
):
    """
    Core analysis loop: extract frames once, then run every requested method
    (PCA/TICA/UMAP) on the same extracted data, running clustering, report
    generation, and CSV export for each.
    """
    if not isinstance(systems, (list, tuple)):
        systems = [systems]

    align_selection = align_selection or selection
    metadata_labels = (
        [item["label"] for item in trajectory_metadata]
        if trajectory_metadata is not None else None
    )
    primary_signature = _validate_selected_atom_order(
        systems, selection, metadata_labels
    )
    primary_align_signature = primary_signature
    if align_selection != selection:
        primary_align_signature = _validate_selected_atom_order(
            systems, align_selection, metadata_labels
        )

    # -- Extract frames once, reuse for all methods --------------------------
    print(f"\n📐  Extracting frames  (selection: '{selection}') …")
    positions, rep_ids, ref_pos, ref_com, frame_indices = _extract_frames(
        systems,
        selection,
        align_selection=align_selection,
        return_reference=True,
        contact_filter_ligand=contact_filter_ligand,
        contact_filter_protein=contact_filter_protein,
        contact_filter_cutoff=contact_filter_cutoff,
    )
    print(f"   Total: {len(positions)} frames × {positions.shape[1]} features")

    projected_payloads = []

    if projected_systems is not None:
        for proj_group, proj_label in zip(projected_systems, projected_label):
            print(f"\n📐  Extracting projected frames  (label: '{proj_label}') …")

            proj_signature = _validate_selected_atom_order(proj_group, selection)
            if proj_signature != primary_signature:
                raise ValueError(
                    f"Projected group '{proj_label}' does not have the same selected "
                    "atom identities/order as the fitted primary systems."
                )
            if align_selection != selection:
                proj_align_signature = _validate_selected_atom_order(
                    proj_group, align_selection
                )
                if proj_align_signature != primary_align_signature:
                    raise ValueError(
                        f"Projected group '{proj_label}' does not have the same alignment "
                        "atom identities/order as the fitted primary systems."
                    )

            proj_positions, proj_rep_ids, proj_frame_indices = _extract_frames(
                proj_group,
                selection,
                align_selection=align_selection,
                ref_pos=ref_pos,
                ref_com=ref_com,
                contact_filter_ligand=contact_filter_ligand,
                contact_filter_protein=contact_filter_protein,
                contact_filter_cutoff=contact_filter_cutoff,
            )

            if proj_positions.shape[1] != positions.shape[1]:
                raise ValueError(
                    "Projected trajectories do not have the same selected feature count "
                    f"({proj_positions.shape[1]} vs {positions.shape[1]})."
                )

            projected_payloads.append((proj_label, proj_positions, proj_rep_ids))

            print(
                f"   Projected total: {len(proj_positions)} frames × "
                f"{proj_positions.shape[1]} features"
            )

    if len(positions) < 3:
        print("❌  Not enough frames for analysis.")
        return

    # Split per-replicate for TICA (must not concatenate across boundaries)
    if rep_ids is not None:
        frame_arrays = [positions[rep_ids == r] for r in np.unique(rep_ids)]
    else:
        frame_arrays = [positions]

    # -- Run each requested method --------------------------------------------
    global CLUSTER_OUTDIR
    for method in methods:
        print(f"\n💠  {method.upper()} …")
        result = _reduce(method, positions, frame_arrays)
        if result is None:
            continue
        if rep_ids is not None:
            result["replicate_ids"] = rep_ids
        result["frame_indices"] = frame_indices
        if trajectory_metadata is not None:
            result["trajectory_metadata"] = trajectory_metadata
        for proj_label, proj_positions, proj_rep_ids in projected_payloads:
            _attach_projection(
                result,
                proj_positions,
                proj_rep_ids,
                method=method,
                label=proj_label,
            )

        out_report   = os.path.join(outdir, f"{prefix}_{method.upper()}.png")
        out_replicates = os.path.join(outdir, f"{prefix}_{method.upper()}_replicates.png")
        out_projected = os.path.join(outdir, f"{prefix}_{method.upper()}_projected.png")

        # Run clustering before report so cluster panels can be included
        if CLUSTER:
            CLUSTER_OUTDIR = os.path.join(outdir, CLUSTER_DIRECTORY_NAME, method.lower())
            result["clustering"] = run_clustering(
                result, systems, method_label=method.upper()
            )

        plot_report(
            result, method=method, output=out_report,
            ncols=REPORT_COLUMNS, save_individual=SAVE_INDIVIDUAL_PANELS,
        )

        # Still save standalone diagnostic plots too
        if PROJECT_PLOT and result.get("projected"):
            plot_projection_overlay(result, method, output=out_projected)

        if REPLICATE_PLOT and rep_ids is not None:
            plot_replicate_embedding(result, method, output=out_replicates)

        if EXPORT_CSV:
            export_csv(result, method, os.path.join(outdir, prefix))


# =============================================================================
# 🧾  SECTION 11 — YAML CONFIGURATION LOADING
# =============================================================================

def _load_yaml_config(path):
    """Load a YAML mapping. PyYAML is required only when --config is used."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "YAML configuration requires PyYAML: pip install pyyaml"
        ) from exc

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("The YAML root must be a mapping/dictionary.")
    return data


def _nested_get(mapping, path, default=None):
    current = mapping
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _config_defaults(cfg):
    """Translate the human-friendly nested YAML schema into argparse defaults.

    See example_config.yaml (shipped alongside this script) for every field
    documented with required/optional status and guidance on when to set it.
    """
    return {
        "topology": _nested_get(cfg, "input.topologies"),
        "trajectory": _nested_get(cfg, "input.trajectories"),
        "trajectory_label": _nested_get(cfg, "input.labels"),
        "trajectory_color": _nested_get(cfg, "input.colors"),
        "selection": _nested_get(cfg, "input.selection", PCA_SEL_MDA),
        "align_selection": _nested_get(cfg, "input.align_selection"),
        "method": _nested_get(cfg, "analysis.methods", METHODS),
        "ncomp": _nested_get(cfg, "analysis.n_components", PCA_NCOMP),
        "var": _nested_get(cfg, "analysis.variance_threshold", PCA_VAR),
        "mode": _nested_get(cfg, "analysis.component_mode", PCA_MODE),
        "lag": _nested_get(cfg, "analysis.tica_lag", TICA_LAG),
        "tica_input_space": _nested_get(cfg, "analysis.input_space.tica", TICA_INPUT_SPACE),
        "umap_input_space": _nested_get(cfg, "analysis.input_space.umap", UMAP_INPUT_SPACE),
        "random_seed": _nested_get(cfg, "analysis.random_seed", RANDOM_SEED),
        "umap_n_components": _nested_get(cfg, "analysis.umap.n_components", UMAP_N_COMPONENTS),
        "umap_n_neighbors": _nested_get(cfg, "analysis.umap.n_neighbors", UMAP_N_NEIGHBORS),
        "umap_min_dist": _nested_get(cfg, "analysis.umap.min_dist", UMAP_MIN_DIST),
        "umap_metric": _nested_get(cfg, "analysis.umap.metric", UMAP_METRIC),
        "interval": _nested_get(cfg, "analysis.frame_stride", MD_INTERVAL),
        "temp": _nested_get(cfg, "analysis.temperature", PCA_TEMP),
        "fel_bins": _nested_get(cfg, "free_energy.bins", PCA_NBINS),
        "fel_sigma": _nested_get(cfg, "free_energy.smoothing_sigma", PCA_SIGMA),
        "prefix": _nested_get(cfg, "output.prefix", OUTPUT_PREFIX),
        "outdir": _nested_get(cfg, "output.directory", "."),
        "cluster_directory": _nested_get(
            cfg, "output.clustering_directory", CLUSTER_DIRECTORY_NAME
        ),
        "palette": _nested_get(cfg, "plotting.palette", _PALETTE),
        "report_columns": _nested_get(cfg, "output.report_columns", REPORT_COLUMNS),
        "no_individual_panels": not _nested_get(
            cfg, "output.save_individual_panels", SAVE_INDIVIDUAL_PANELS
        ),
        "no_csv": not _nested_get(cfg, "output.export_csv", EXPORT_CSV),
        "no_cluster": not _nested_get(cfg, "clustering.enabled", CLUSTER),
        "cluster_method": _nested_get(cfg, "clustering.method", CLUSTER_METHOD),
        "kmax": _nested_get(cfg, "clustering.kmax", CLUSTER_KMAX),
        "cluster_inertia_threshold": _nested_get(
            cfg, "clustering.kmeans.inertia_threshold", CLUSTER_INERTIA_THRESHOLD
        ),
        "cluster_variance_threshold": _nested_get(
            cfg, "clustering.variance_threshold", CLUSTER_VARIANCE_THRESHOLD
        ),
        "gmm_covariance_type": _nested_get(
            cfg, "clustering.gmm.covariance_type", GMM_COVARIANCE_TYPE
        ),
        "gmm_reg_covar": _nested_get(
            cfg, "clustering.gmm.reg_covar", GMM_REG_COVAR
        ),
        "gmm_bic_delta_stop": _nested_get(
            cfg, "clustering.gmm.bic_delta_stop", GMM_BIC_DELTA_STOP,
        ),
        "gmm_n_init": _nested_get(cfg, "clustering.gmm.n_init", GMM_N_INIT),
        "hdbscan_min_cluster_size": _nested_get(
            cfg, "clustering.hdbscan.min_cluster_size", HDBSCAN_MIN_CLUSTER_SIZE
        ),
        "hdbscan_min_samples": _nested_get(
            cfg, "clustering.hdbscan.min_samples", HDBSCAN_MIN_SAMPLES
        ),
        "hdbscan_metric": _nested_get(
            cfg, "clustering.hdbscan.metric", HDBSCAN_METRIC
        ),
        "hdbscan_selection_method": _nested_get(
            cfg, "clustering.hdbscan.cluster_selection_method",
            HDBSCAN_CLUSTER_SELECTION_METHOD,
        ),
        "silhouette": _nested_get(
            cfg, "clustering.compute_silhouette", CLUSTER_SILHOUETTE
        ),
        "silhouette_sample": _nested_get(
            cfg, "clustering.silhouette_sample", CLUSTER_SILHOUETTE_SAMPLE
        ),
        "cluster_select_by": _nested_get(
            cfg, "clustering.select_by", CLUSTER_SELECT_BY
        ),
        "cluster_use_umap": _nested_get(
            cfg, "clustering.use_umap", CLUSTER_USE_UMAP
        ),
        "no_centroids": not _nested_get(
            cfg, "clustering.extract_centroids", CLUSTER_EXTRACT_CENTROIDS
        ),
        "centroid_ligand_selection": _nested_get(
            cfg, "clustering.centroid_ligand_selection", CLUSTER_CENTROID_LIGAND_SELECTION
        ),
        "ellipse_level": _nested_get(
            cfg, "plotting.ellipse_level", REPLICATE_ELLIPSE_LEVEL
        ),
        "no_replicate_plot": not _nested_get(
            cfg, "plotting.save_replicate_plot", REPLICATE_PLOT
        ),
        "no_project_plot": not _nested_get(
            cfg, "plotting.save_projection_plot", PROJECT_PLOT
        ),
        "contact_filter_ligand": _nested_get(cfg, "contact_filter.ligand"),
        "contact_filter_protein": _nested_get(
            cfg, "contact_filter.protein", "protein and not name H*"
        ),
        "contact_filter_cutoff": _nested_get(cfg, "contact_filter.cutoff", 4.0),
        "project_groups": _nested_get(cfg, "projection.groups", []),
    }


# =============================================================================
# 🖥️  SECTION 12 — CLI
# =============================================================================

def _build_parser(defaults):
    p = argparse.ArgumentParser(
        prog="quick_pca.py",
        description="QuickPCA v3.0 — MDAnalysis essential dynamics with YAML support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.set_defaults(**defaults)
    p.add_argument("--config", help="YAML configuration file.")
    p.add_argument("-t", "--topology", nargs="+",
                   help="One shared topology or one topology per trajectory.")
    p.add_argument("-r", "--trajectory", nargs="+",
                   help="Primary trajectory file(s).")
    p.add_argument("--trajectory-label", nargs="+",
                   help="One legend label per primary trajectory.")
    p.add_argument("--trajectory-color", nargs="+",
                   help="One Matplotlib-compatible color per primary trajectory.")
    p.add_argument("--project-topology", nargs="+", action="append",
                   help="Topology group for one projected trajectory group.")
    p.add_argument("--project-trajectory", nargs="+", action="append",
                   help="Projected trajectory group; repeat for multiple systems.")
    p.add_argument("--project-label", action="append",
                   help="Label for each projected trajectory group.")
    p.add_argument("--align-selection",
                   help="Alignment selection; defaults to the PCA selection.")
    p.add_argument("-s", "--selection", help="MDAnalysis atom selection.")
    p.add_argument("-m", "--method", nargs="+", choices=["pca", "tica", "umap"])
    p.add_argument("--ncomp", type=int)
    p.add_argument("--var", type=float)
    p.add_argument("--mode", choices=["count", "variance"])
    p.add_argument("--lag", type=int)
    p.add_argument("--tica-input-space", choices=["coordinates", "pca"])
    p.add_argument("--umap-input-space", choices=["coordinates", "pca"])
    p.add_argument("--random-seed", type=int)
    p.add_argument("--umap-n-components", type=int)
    p.add_argument("--umap-n-neighbors", type=int)
    p.add_argument("--umap-min-dist", type=float)
    p.add_argument("--umap-metric")
    p.add_argument("--interval", type=int)
    p.add_argument("--temp", type=float)
    p.add_argument("--fel-bins", type=int)
    p.add_argument("--fel-sigma", type=float)
    p.add_argument("--prefix")
    p.add_argument("--outdir")
    p.add_argument("--cluster-directory")
    p.add_argument("--palette", nargs="+")
    p.add_argument("--report-columns", type=int,
                   help="Number of subplot columns in the combined report.")
    p.add_argument("--no-individual-panels", action="store_true")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--no-cluster", action="store_true")
    p.add_argument("--cluster-method", choices=["kmeans", "gmm", "hdbscan"])
    p.add_argument("--kmax", type=int)
    p.add_argument("--cluster-inertia-threshold", type=float)
    p.add_argument(
        "--cluster-variance-threshold", type=float,
        help="Use the smallest leading component set reaching this cumulative variance fraction.",
    )
    p.add_argument("--gmm-covariance-type", choices=["full", "tied", "diag", "spherical"])
    p.add_argument("--gmm-reg-covar", type=float)
    p.add_argument("--gmm-bic-delta-stop", type=float)
    p.add_argument("--gmm-n-init", type=int)
    p.add_argument("--hdbscan-min-cluster-size", type=int)
    p.add_argument("--hdbscan-min-samples", type=int)
    p.add_argument("--hdbscan-metric")
    p.add_argument(
        "--hdbscan-selection-method", choices=["eom", "leaf"]
    )
    p.add_argument("--silhouette", action="store_true")
    p.add_argument("--silhouette-sample", type=int)
    p.add_argument("--cluster-select-by", choices=["elbow", "silhouette"])
    p.add_argument("--cluster-use-umap", action="store_true")
    p.add_argument("--no-centroids", action="store_true")
    p.add_argument("--centroid-ligand-selection",
                   help="MDAnalysis selection for a ligand to re-image near the "
                        "protein before writing centroid PDBs. Omit for protein-only "
                        "or non-periodic systems.")
    p.add_argument("--ellipse-level", type=float)
    p.add_argument("--no-replicate-plot", action="store_true")
    p.add_argument("--no-project-plot", action="store_true")
    p.add_argument("--contact-filter-ligand")
    p.add_argument("--contact-filter-protein")
    p.add_argument("--contact-filter-cutoff", type=float)
    return p


def main_cli():
    """Standalone MDAnalysis entry point with YAML plus optional CLI overrides."""
    global PCA_NCOMP, PCA_VAR, PCA_MODE, TICA_LAG, MD_INTERVAL, PCA_TEMP
    global PCA_NBINS, PCA_SIGMA, TICA_INPUT_SPACE, UMAP_INPUT_SPACE, RANDOM_SEED
    global UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_METRIC
    global EXPORT_CSV, CLUSTER, CLUSTER_KMAX, CLUSTER_VARIANCE_THRESHOLD
    global CLUSTER_EXTRACT_CENTROIDS, CLUSTER_CENTROID_LIGAND_SELECTION
    global CLUSTER_SILHOUETTE, CLUSTER_SILHOUETTE_SAMPLE, CLUSTER_SELECT_BY
    global CLUSTER_METHOD, GMM_COVARIANCE_TYPE, GMM_REG_COVAR
    global GMM_BIC_DELTA_STOP, GMM_N_INIT, CLUSTER_INERTIA_THRESHOLD
    global HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, HDBSCAN_METRIC
    global HDBSCAN_CLUSTER_SELECTION_METHOD, CLUSTER_DIRECTORY_NAME
    global CLUSTER_USE_UMAP, REPLICATE_ELLIPSE_LEVEL, PROJECT_PLOT
    global REPLICATE_PLOT, REPORT_COLUMNS, SAVE_INDIVIDUAL_PANELS
    global _PALETTE

    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config")
    probe_args, _ = config_probe.parse_known_args()
    config = _load_yaml_config(probe_args.config) if probe_args.config else {}
    defaults = _config_defaults(config)
    parser = _build_parser(defaults)
    args = parser.parse_args()

    PCA_NCOMP = args.ncomp
    PCA_VAR = args.var
    PCA_MODE = args.mode
    TICA_LAG = args.lag
    TICA_INPUT_SPACE = args.tica_input_space
    UMAP_INPUT_SPACE = args.umap_input_space
    RANDOM_SEED = args.random_seed
    UMAP_N_COMPONENTS = args.umap_n_components
    UMAP_N_NEIGHBORS = args.umap_n_neighbors
    UMAP_MIN_DIST = args.umap_min_dist
    UMAP_METRIC = args.umap_metric
    MD_INTERVAL = args.interval
    PCA_TEMP = args.temp
    PCA_NBINS = args.fel_bins
    PCA_SIGMA = args.fel_sigma
    EXPORT_CSV = not args.no_csv
    CLUSTER = not args.no_cluster
    CLUSTER_METHOD = args.cluster_method
    CLUSTER_KMAX = args.kmax
    CLUSTER_INERTIA_THRESHOLD = args.cluster_inertia_threshold
    CLUSTER_VARIANCE_THRESHOLD = args.cluster_variance_threshold
    if not 0.0 < CLUSTER_VARIANCE_THRESHOLD <= 1.0:
        parser.error("--cluster-variance-threshold must be > 0 and <= 1.")
    GMM_COVARIANCE_TYPE = args.gmm_covariance_type
    GMM_REG_COVAR = args.gmm_reg_covar
    GMM_BIC_DELTA_STOP = args.gmm_bic_delta_stop
    GMM_N_INIT = args.gmm_n_init
    HDBSCAN_MIN_CLUSTER_SIZE = args.hdbscan_min_cluster_size
    HDBSCAN_MIN_SAMPLES = args.hdbscan_min_samples
    HDBSCAN_METRIC = args.hdbscan_metric
    HDBSCAN_CLUSTER_SELECTION_METHOD = args.hdbscan_selection_method
    CLUSTER_DIRECTORY_NAME = args.cluster_directory
    CLUSTER_SILHOUETTE = args.silhouette
    CLUSTER_SILHOUETTE_SAMPLE = max(100, args.silhouette_sample)
    CLUSTER_SELECT_BY = args.cluster_select_by
    CLUSTER_USE_UMAP = args.cluster_use_umap
    CLUSTER_EXTRACT_CENTROIDS = not args.no_centroids
    CLUSTER_CENTROID_LIGAND_SELECTION = args.centroid_ligand_selection
    REPLICATE_ELLIPSE_LEVEL = args.ellipse_level
    REPLICATE_PLOT = not args.no_replicate_plot
    PROJECT_PLOT = not args.no_project_plot
    REPORT_COLUMNS = max(1, args.report_columns)
    SAVE_INDIVIDUAL_PANELS = not args.no_individual_panels
    _PALETTE = list(args.palette)

    if PCA_NBINS < 2:
        parser.error("--fel-bins must be at least 2.")
    if PCA_SIGMA < 0:
        parser.error("--fel-sigma must be non-negative.")
    if not 0.0 < CLUSTER_INERTIA_THRESHOLD < 1.0:
        parser.error("--cluster-inertia-threshold must be between 0 and 1.")
    if UMAP_N_COMPONENTS < 2 or UMAP_N_NEIGHBORS < 2 or UMAP_MIN_DIST < 0:
        parser.error("UMAP requires n_components >= 2, n_neighbors >= 2, and min_dist >= 0.")
    if GMM_N_INIT < 1:
        parser.error("--gmm-n-init must be at least 1.")
    if TICA_INPUT_SPACE not in {"coordinates", "pca"}:
        parser.error("analysis.input_space.tica must be 'coordinates' or 'pca'.")
    if UMAP_INPUT_SPACE not in {"coordinates", "pca"}:
        parser.error("analysis.input_space.umap must be 'coordinates' or 'pca'.")
    if not _PALETTE:
        parser.error("plotting.palette must contain at least one color.")
    if not CLUSTER_DIRECTORY_NAME or os.path.isabs(CLUSTER_DIRECTORY_NAME):
        parser.error("--cluster-directory must be a non-empty relative directory name.")

    try:
        import MDAnalysis as mda
    except ImportError:
        print("❌  MDAnalysis not available. pip install MDAnalysis")
        sys.exit(1)

    if not args.topology:
        parser.error("Explicit topology input is required: set input.topologies or --topology.")
    if not args.trajectory:
        parser.error("Explicit trajectory input is required: set input.trajectories or --trajectory.")

    trajs = list(args.trajectory)
    topologies = _expand_topologies(args.topology, trajs)
    labels = _normalise_optional_list(
        args.trajectory_label, len(trajs), "trajectory-label",
        lambda i: f"Trajectory {i + 1}",
    )
    colors = _normalise_optional_list(
        args.trajectory_color, len(trajs), "trajectory-color",
        lambda i: _PALETTE[i % len(_PALETTE)],
    )
    systems = [mda.Universe(top, traj) for top, traj in zip(topologies, trajs)]
    trajectory_metadata = [
        {"label": label, "color": color,
         "topology": top, "trajectory": traj}
        for label, color, top, traj in zip(labels, colors, topologies, trajs)
    ]

    projected_systems = None
    projected_labels = None
    yaml_groups = args.project_groups or []

    if args.project_trajectory:
        yaml_groups = []
        top_groups = args.project_topology
        if top_groups is None:
            if len(set(topologies)) != 1:
                parser.error("Supply projected topologies when primary topologies differ.")
            top_groups = [[topologies[0]] for _ in args.project_trajectory]
        labels = args.project_label or [
            f"Projected {i + 1}" for i in range(len(args.project_trajectory))
        ]
        if len(top_groups) != len(args.project_trajectory) or len(labels) != len(args.project_trajectory):
            parser.error("Projected topology, trajectory, and label group counts must match.")
        yaml_groups = [
            {"topologies": tops, "trajectories": trajs, "label": label}
            for tops, trajs, label in zip(top_groups, args.project_trajectory, labels)
        ]

    if yaml_groups:
        projected_systems, projected_labels = [], []
        for i, group in enumerate(yaml_groups, start=1):
            if not isinstance(group, dict):
                raise ValueError("Each projection.groups entry must be a mapping.")
            traj_group = group.get("trajectories") or []
            if not traj_group:
                raise ValueError(f"Projection group {i} has no trajectories.")
            top_group = group.get("topologies")
            if top_group is None:
                if len(set(topologies)) != 1:
                    raise ValueError(
                        "Projected topologies are required when primary topologies differ."
                    )
                top_group = [topologies[0]]
            expanded = _expand_topologies(
                top_group, traj_group, option_name="projection.groups.topologies"
            )
            projected_systems.append([
                mda.Universe(top, traj) for top, traj in zip(expanded, traj_group)
            ])
            projected_labels.append(group.get("label", f"Projected {i}"))

    os.makedirs(args.outdir, exist_ok=True)
    start = time.time()
    print("🔌  Backend: MDAnalysis")
    if args.config:
        print(f"🧾  Config: {args.config}")
    _run_pipeline(
        systems, args.selection, args.method, args.prefix, args.outdir,
        projected_systems=projected_systems,
        projected_label=projected_labels,
        contact_filter_ligand=args.contact_filter_ligand,
        contact_filter_protein=args.contact_filter_protein,
        contact_filter_cutoff=args.contact_filter_cutoff,
        align_selection=args.align_selection or args.selection,
        trajectory_metadata=trajectory_metadata,
    )
    elapsed = time.time() - start
    h, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n🕰️  Total: {int(h)}h {int(minutes)}m {int(seconds)}s")


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    main_cli()
