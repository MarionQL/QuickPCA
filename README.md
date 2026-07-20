# 💠 QuickPCA: welcome to the universe of eigenvectors!

# QuickPCA: PCA, TICA, UMAP, clustering, and free-energy landscapes for MD trajectories

<img 
    src="https://github.com/TheVisualHub/VisualFactory/blob/main/assets/quickPCA_logo.png" 
    alt="QuickPCA Logo" 
    width="800">

QuickPCA is a lightweight Python tool for Essential Dynamics Analysis of molecular dynamics trajectories in PyMOL. It automatically detects and loads MD trajectories, performs Principal Component Analysis, and generates a publication-ready report featuring free-energy landscapes, residue cross-correlation maps, explained variance profiles and principal component projections.

Molecular dynamics trajectories have a massive amount of data. If your protein has 1,000 atoms, each frame has 3,000 coordinates (X, Y, Z). If you have 10,000 frames, that is 30 million data points! PCA analyzes this massive dataset and reduces the dimensions. It figures out which atomic movements are just random "noise" and which are the "signals" capturing functional-relevant motions. This algorithm compresses thousands of dimensions down into just two (PC1 and PC2) while preserving the most important information.

Unlike common PCA approaches, which construct and diagonalize the covariance matrix, quickPCA performs SVD decomposition using scikit-learn directly on the (n_frames × 3N) data matrix. This avoids the costly step of diagonalizing the covariance matrix, making the approach faster and more numerically stable, while producing identical principal components. The residue cross-correlation matrix is subsequently recovered analytically from the PCA eigenvectors and eigenvalues, without revisiting the raw trajectory data.

## What PCA does

An MD frame containing *N* selected atoms is represented by `3N` Cartesian coordinates. Even a C-alpha-only trajectory can therefore contain hundreds or thousands of features. Principal component analysis (PCA) finds orthogonal directions through this coordinate space that explain decreasing amounts of positional variance.

- **PC1** explains the largest possible fraction of coordinate variance.
- **PC2** explains the largest remaining fraction while being orthogonal to PC1.
- Later PCs describe progressively smaller-amplitude motions.
- A frame's **projection** is its coordinate along each PC.
- A PC's **loading/eigenvector** describes how each selected atom moves along that PC.

Before PCA, frames must be aligned. Otherwise, translation and rotation of the entire molecule can dominate the apparent variance. This script aligns every frame to the first sampled frame using `input.align_selection`, then performs the analysis using `input.selection`.

PCA identifies large-amplitude correlated motions; it does **not** prove that a PC is a kinetic process, a reaction coordinate, or a thermodynamic state. Results also depend on the atom selection, alignment selection, sampling, and whether trajectories are equilibrated and converged.

### Count mode versus variance mode

- `component_mode: count` calculates exactly `n_components` PCs (or the maximum mathematically available). It is fast and suitable when you know how many components you need.
- `component_mode: variance` calculates enough PCs to reach `variance_threshold`. It uses full SVD and can require considerably more memory and time.

Use count mode for exploratory work or very large datasets. Use variance mode when downstream analysis must retain a stated fraction of PCA variance.

## Related reduction methods

### TICA

Time-lagged independent component analysis (TICA) emphasizes slowly decorrelating changes rather than the largest positional fluctuations. Its axes are called independent components (ICs). `analysis.tica_lag` is measured in **sampled frames after applying `frame_stride`**, not in raw frames or physical time.

TICA is useful when metastable-state separation and kinetics matter. Test multiple lag times and look for stable implied timescales; one arbitrary lag does not validate a kinetic model. Replicate boundaries are preserved by the script, which prevents transitions from the end of one replicate to the beginning of another.

### UMAP

UMAP is a nonlinear neighborhood-preserving embedding. In this script, UMAP is trained on an upstream PCA projection rather than raw Cartesian coordinates. It is primarily useful for visualization and exploratory grouping.

Distances, cluster shapes, and apparent gaps in UMAP should not automatically be interpreted as physical barriers or kinetics. UMAP may operate on aligned Cartesian coordinates or PCA coordinates, and its dimensionality, neighborhood size, minimum distance, metric, and random seed are configurable.

## What clustering does

Clustering groups frames with similar reduced coordinates. It does not discover a uniquely correct set of conformational states: the result depends on the reduction method, retained dimensions, distance model, hyperparameters, and sampling.

For PCA and TICA, the script clusters on the smallest number of leading components that reaches `clustering.variance_threshold`, with at least two dimensions when available. For UMAP, which has no explained-variance ratios, it uses every available embedding dimension (currently two).

### K-means

K-means assigns every frame to the nearest centroid and favors compact, roughly spherical, similarly sized clusters.

Use it when clusters look compact and you want a simple, reproducible partition. Avoid treating it as definitive when states overlap strongly, have unequal density, or contain substantial transition/noise regions.

The script tests `k = 2...kmax`. Its elbow rule stops increasing *k* when the relative inertia improvement falls below a fixed **15% constant**. If `compute_silhouette: true` and `select_by: silhouette`, it instead selects the tested *k* with the largest sampled silhouette score.

### Gaussian mixture model (GMM)

A GMM represents the data as overlapping Gaussian distributions. It can model ellipsoidal clusters and provides soft probabilistic structure internally, although this script exports hard labels from `predict()`.

Use GMM when conformational populations overlap or have different covariance shapes. `full` covariance is flexible but parameter-heavy; `diag`, `tied`, or `spherical` covariance can be more stable with limited data or many dimensions.

The script tests `k = 2...kmax` and normally considers BIC. Its custom early-stop rule chooses the simpler model as soon as the BIC improvement to the next *k* falls below `gmm.bic_delta_stop`; this can differ from choosing the global minimum BIC.

### HDBSCAN

HDBSCAN finds dense regions without requiring *k* and labels sparse or transitional points as noise (`-1`).

Use it when clusters have irregular shapes, noise is scientifically meaningful, or a forced assignment is undesirable. It can return no clusters if `min_cluster_size` or `min_samples` is too strict. The supplied script fixes Euclidean distance and the `eom` selection method.

### `clustering.use_umap`

When true, the current script performs a **second, separate clustering run on a new UMAP embedding**, in addition to clustering the original PCA/TICA/UMAP coordinates. It is not merely a prettier visualization. Labels and centroid structures are produced for both runs. Because nonlinear embedding can alter geometry, treat UMAP-derived clusters as exploratory.

## Free-energy landscape (FEL)

The script estimates a two-dimensional density from the first two components, smooths it, and calculates

$$F(x,y)=-RT\ln P(x,y),$$

then shifts the minimum to zero. The result is a population-derived apparent free-energy surface in kJ/mol. It is only reliable where sampling is adequate. Histogram binning, smoothing, correlated frames, and lack of convergence can change the surface. Histogram bins and smoothing sigma are configurable under `free_energy`.

## Installation

Python 3.9+ is recommended. Install the base dependencies:

```bash
python -m pip install numpy scipy matplotlib scikit-learn MDAnalysis PyYAML
```

Install only the optional packages needed for requested methods:

```bash
python -m pip install deeptime      # TICA
python -m pip install umap-learn    # UMAP or clustering.use_umap
python -m pip install hdbscan       # HDBSCAN clustering
```

For reproducible distribution, provide a tested `requirements.txt` or environment file with pinned versions. `KMeans(n_init="auto")` requires a reasonably recent scikit-learn release.

## Input requirements

- One topology may be shared by all primary trajectories, or one topology must be supplied per trajectory.
- Every analysis selection must contain the same atoms in the same order in all primary and projected systems.
- Projected systems must use the same analysis and alignment atom identities/order as the fitted primary systems.
- Coordinates and cutoff distances are assumed to be in MDAnalysis units (normally angstroms).
- For meaningful combined analysis, replicas should describe the same molecular system and comparable ensembles.
- File paths are interpreted relative to the directory from which the command is run, not relative to the YAML file.

The atom-order check uses `(resname, resid, atom name)` and does not include chain ID, segment ID, insertion code, or atom index. Systems with repeated residue numbering across chains therefore require extra care.

## Quick start

1. Copy `example_config.yaml` and replace the example paths and selections.
2. Run:

```bash
python quick_pca.py --config example_config.yaml
```

Command-line flags can override most YAML values. Boolean CLI flags are asymmetric (`--no-csv`, `--no-cluster`, and so on), so YAML is the clearer interface for reusable workflows.

## How to choose the main settings

### Atom selection

- `protein and name CA`: common for global protein-domain motion; low dimensional and relatively easy to interpret.
- `protein and backbone`: captures finer backbone motion at higher computational cost.
- A stable domain or residue range: useful when the scientific question is localized.
- Including a flexible ligand or side chains: possible, but all systems must have identical selected atoms and ordering.

The alignment selection may be narrower than the analysis selection. For example, align on a stable domain while analyzing the whole protein. Avoid aligning on highly flexible atoms whose motion you intend to measure.

### Frame stride

Use `frame_stride` to reduce redundant, highly correlated frames and memory use. The script loads all retained aligned coordinates into RAM as float64, approximately `8 × retained_frames × 3 × selected_atoms` bytes before downstream copies. Striding does not replace convergence analysis.

### Combining replicas

The script fits one shared PCA/TICA/UMAP model across all primary trajectories and weights every retained frame equally. A longer trajectory therefore contributes more weight. Use equal sampled lengths if equal replicate weighting is intended.

### Projection

Projection places secondary trajectories on axes fitted only from the primary trajectories. Use it to compare mutant versus wild type, apo versus bound, or held-out simulations without refitting the basis. PCA projection is linear; TICA uses the fitted model; UMAP uses an approximate out-of-sample transform and should be interpreted cautiously.

### Contact filtering

The optional contact filter keeps a frame when the **median across ligand atoms of each atom's minimum protein distance** is at or below the cutoff. This is stricter and different from “any ligand atom is within 4 Å.”

Filtering can be helpful for pocket-focused analyses, but it changes the sampled ensemble. For TICA it can also remove intermediate frames and make originally nonconsecutive retained frames appear consecutive to the estimator; kinetic interpretations are therefore unsafe without modifying the implementation to preserve gaps as separate trajectory segments.

## YAML reference

The companion `example_config.yaml` contains every YAML key recognized by the supplied script, including conditional fields and comments describing required status. Important relationships:

- `input.trajectories` is required for normal MD analysis. Without it, the script searches the working directory for one supported trajectory or analyzes one topology frame.
- `input.topologies` and `input.trajectories` are required; automatic discovery has been removed.
- `analysis.n_components` is used in `count` mode and as a practical component count for TICA; `analysis.variance_threshold` is used in `variance` mode.
- GMM fields matter only for `clustering.method: gmm`; HDBSCAN fields only for `hdbscan`.
- Silhouette settings affect K-means only.
- `projection.groups` and `contact_filter` are optional features.

Unknown YAML keys are silently ignored. Misspellings therefore do not produce an error.

## Outputs

For each requested method, the script can create:

- `<prefix>_<METHOD>.png`: combined report.
- `<prefix>_<METHOD>_replicates.png`: replicate overlay when multiple primary trajectories are supplied.
- `<prefix>_<METHOD>_projected.png`: primary-versus-projected overlay.
- `<prefix>_<METHOD>_variance.csv`: component variance and, for TICA, timescales.
- `<prefix>_<METHOD>_loadings.csv`: PCA/TICA eigenvectors.
- `<prefix>_<METHOD>_<label>_projected.csv`: projected coordinates.
- `individual_panels/`: standalone report panels.
- `clustering/<method>/`: model-selection plots/tables, labels, cluster plots, and centroid PDBs.

The cluster label CSV's `frame` column is the row number in the retained dataset, **not necessarily the original trajectory frame**. Original frame indices are used internally for centroid extraction but are not exported in the label CSV. In multi-replicate output, the label CSV also omits replicate identity.

Centroid PDBs contain the entire raw topology frame, not the aligned coordinates shown to PCA. The representative frame is chosen as the observed point nearest the cluster center in clustering space.

## Interpreting results responsibly

- Check replicate overlap and report each replicate separately as well as combined.
- Confirm that conclusions are stable to atom selection, alignment selection, stride, component count, and clustering settings.
- Do not call clusters metastable states based on PCA plus K-means alone.
- For kinetics, validate TICA lag dependence and use a workflow designed for trajectory continuity and state-model validation.
- Inspect centroid structures and cluster populations; numerical labels have no inherent ordering or physical meaning.

Released under the MIT License. If QuickPCA is used in any capacity that contributes to results presented in a publication, thesis, report, or any other form of scholarly or professional work, appropriate citation of QuickPCA is strongly encouraged.
