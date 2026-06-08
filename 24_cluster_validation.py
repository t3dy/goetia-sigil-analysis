"""
Script 24: Cluster Count Validation

The original pipeline fixed n_clusters=8 without validation.
This script tests k=2 through k=15 using three indices:
  - Silhouette score (higher = better separated)
  - Davies-Bouldin index (lower = better)
  - Gap statistic (gap between within-cluster dispersion and null distribution)

Uses the normalized sigils from 23_adaptive_preprocessing.py as input
via re-extracted features, OR falls back to the original features.json
if the normalized features aren't available yet.

Outputs:
  - cluster_validation.json  (scores for each k)
  - cluster_validation.png   (elbow + silhouette + DB curves)
  - Prints recommended k with reasoning
"""

import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

OUTDIR = Path(r"C:\Dev\GoetiaRevEng")

# ── Load feature matrix (prefer re-extracted from normalized sigils) ──────────

features_file = OUTDIR / "features.json"
if not features_file.exists():
    # Try old path
    features_file = Path(r"C:\Users\PC\Downloads\goetia_analysis\features.json")

with open(features_file) as f:
    features = json.load(f)

skel_file = OUTDIR / "skeleton_analysis.json"
if not skel_file.exists():
    skel_file = Path(r"C:\Users\PC\Downloads\goetia_analysis\skeleton_analysis.json")
skel_map = {}
if skel_file.exists():
    with open(skel_file) as f:
        skel_map = {s["id"]: s for s in json.load(f)}

hough_file = OUTDIR / "hough_analysis.json"
if not hough_file.exists():
    hough_file = Path(r"C:\Users\PC\Downloads\goetia_analysis\hough_analysis.json")
hough_map = {}
if hough_file.exists():
    with open(hough_file) as f:
        hough_map = {h["id"]: h for h in json.load(f)}

# Build feature vectors (same construction as 06_clustering.py)
feature_vectors = []
sigil_ids = []

for feat in features:
    sid = feat["id"]
    vec = [
        feat["ink_ratio"],
        feat["aspect_ratio"],
        feat["compactness"],
        feat["fractal_dimension"],
        feat["horizontal_symmetry"],
        feat["vertical_symmetry"],
    ]
    vec.extend(feat["radial_profile"])
    vec.extend(feat["quadrant_density"])

    if sid in skel_map:
        s = skel_map[sid]
        vec.extend([
            s["connected_components"] / 10.0,
            s["junctions"] / 100.0,
            s["endpoints"] / 50.0,
            s["holes"] / 10.0,
            s["junction_endpoint_ratio"],
        ])
    else:
        vec.extend([0, 0, 0, 0, 0])

    if sid in hough_map:
        h = hough_map[sid]
        vec.extend([h["n_lines"] / 100.0, h["n_circles"] / 20.0,
                    h["line_circle_ratio"] / 50.0])
        ah = np.array(h["angle_histogram"], dtype=float)
        if ah.sum() > 0:
            ah /= ah.sum()
        vec.extend(ah.tolist())
    else:
        vec.extend([0, 0, 0] + [0] * 12)

    feature_vectors.append(vec)
    sigil_ids.append(sid)

X = np.array(feature_vectors)
print(f"Feature matrix: {X.shape[0]} × {X.shape[1]}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 10 PCA components before clustering (stabilizes distances with 38D/72 samples)
n_components = min(10, X_scaled.shape[0] - 1, X_scaled.shape[1])
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA to {n_components} components: {pca.explained_variance_ratio_.sum():.1%} variance retained")

# ── Gap statistic ──────────────────────────────────────────────────────────────

def gap_statistic(X, linkage_matrix, k_range, n_refs=20, random_state=42):
    """
    Compare within-cluster dispersion to a uniform reference distribution.
    Returns gap values and std errors for each k in k_range.
    """
    rng = np.random.default_rng(random_state)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    def wk(X, labels):
        total = 0.0
        for cl in np.unique(labels):
            pts = X[labels == cl]
            if len(pts) < 2:
                continue
            d = squareform(pdist(pts))
            total += d.sum() / (2 * len(pts))
        return total

    log_wk_real = []
    for k in k_range:
        labels = fcluster(linkage_matrix, k, criterion='maxclust')
        log_wk_real.append(np.log(wk(X, labels) + 1e-10))

    log_wk_refs = np.zeros((len(k_range), n_refs))
    for b in range(n_refs):
        X_ref = rng.uniform(mins, maxs, size=X.shape)
        lm_ref = linkage(X_ref, method='ward')
        for i, k in enumerate(k_range):
            labels_ref = fcluster(lm_ref, k, criterion='maxclust')
            log_wk_refs[i, b] = np.log(wk(X_ref, labels_ref) + 1e-10)

    gaps = log_wk_refs.mean(axis=1) - np.array(log_wk_real)
    sdk = log_wk_refs.std(axis=1) * np.sqrt(1 + 1.0 / n_refs)
    return gaps, sdk


# ── Run validation ─────────────────────────────────────────────────────────────

k_range = list(range(2, 16))
linkage_matrix = linkage(X_pca, method='ward')

silhouette_scores = []
db_scores         = []
inertia_vals      = []

for k in k_range:
    labels = fcluster(linkage_matrix, k, criterion='maxclust')
    sil = silhouette_score(X_pca, labels)
    db  = davies_bouldin_score(X_pca, labels)
    silhouette_scores.append(sil)
    db_scores.append(db)

    # Ward inertia: sum of within-cluster variances
    inertia = sum(
        np.var(X_pca[labels == cl], axis=0).sum() * np.sum(labels == cl)
        for cl in np.unique(labels)
    )
    inertia_vals.append(inertia)
    print(f"  k={k:2d}  silhouette={sil:.3f}  DB={db:.3f}  inertia={inertia:.1f}")

print("Computing gap statistic (this takes ~30 seconds)...")
gaps, gap_se = gap_statistic(X_pca, linkage_matrix, k_range, n_refs=20)

# ── Recommend k ───────────────────────────────────────────────────────────────

# Gap rule: first k where gap(k) >= gap(k+1) - se(k+1)
gap_k = None
for i in range(len(k_range) - 1):
    if gaps[i] >= gaps[i+1] - gap_se[i+1]:
        gap_k = k_range[i]
        break

sil_k  = k_range[np.argmax(silhouette_scores)]
db_k   = k_range[np.argmin(db_scores)]

print(f"\n=== Recommendation ===")
print(f"  Silhouette peak:  k={sil_k}  (score={max(silhouette_scores):.3f})")
print(f"  Davies-Bouldin min: k={db_k}  (score={min(db_scores):.3f})")
print(f"  Gap statistic:    k={gap_k}")
votes = [sil_k, db_k, gap_k]
from collections import Counter
best_k = Counter(votes).most_common(1)[0][0]
print(f"  Consensus (majority vote): k={best_k}")
print(f"  Original fixed value: k=8")

# ── Save results ──────────────────────────────────────────────────────────────

records = []
for i, k in enumerate(k_range):
    records.append({
        "k":              k,
        "silhouette":     round(silhouette_scores[i], 4),
        "davies_bouldin": round(db_scores[i], 4),
        "inertia":        round(inertia_vals[i], 2),
        "gap":            round(float(gaps[i]), 4),
        "gap_se":         round(float(gap_se[i]), 4),
    })

with open(OUTDIR / "cluster_validation.json", "w") as f:
    json.dump({
        "n_samples":          X.shape[0],
        "n_features_raw":     X.shape[1],
        "n_pca_components":   n_components,
        "pca_variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
        "recommended_k": {
            "silhouette": sil_k,
            "davies_bouldin": db_k,
            "gap_statistic": gap_k,
            "consensus": best_k,
            "original_fixed": 8,
        },
        "scores_by_k": records,
    }, f, indent=2)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

axes[0,0].plot(k_range, silhouette_scores, 'o-', color='steelblue')
axes[0,0].axvline(sil_k, color='steelblue', linestyle='--', alpha=0.5, label=f'peak k={sil_k}')
axes[0,0].axvline(8, color='red', linestyle=':', alpha=0.4, label='original k=8')
axes[0,0].set_title("Silhouette Score (higher = better)")
axes[0,0].set_xlabel("Number of clusters k")
axes[0,0].legend(fontsize=8)

axes[0,1].plot(k_range, db_scores, 'o-', color='darkorange')
axes[0,1].axvline(db_k, color='darkorange', linestyle='--', alpha=0.5, label=f'min k={db_k}')
axes[0,1].axvline(8, color='red', linestyle=':', alpha=0.4, label='original k=8')
axes[0,1].set_title("Davies-Bouldin Index (lower = better)")
axes[0,1].set_xlabel("Number of clusters k")
axes[0,1].legend(fontsize=8)

axes[1,0].plot(k_range, inertia_vals, 'o-', color='mediumseagreen')
axes[1,0].axvline(8, color='red', linestyle=':', alpha=0.4, label='original k=8')
axes[1,0].set_title("Within-Cluster Variance (elbow)")
axes[1,0].set_xlabel("Number of clusters k")
axes[1,0].legend(fontsize=8)

axes[1,1].errorbar(k_range, gaps, yerr=gap_se, fmt='o-', color='purple', capsize=4)
if gap_k:
    axes[1,1].axvline(gap_k, color='purple', linestyle='--', alpha=0.5, label=f'gap k={gap_k}')
axes[1,1].axvline(8, color='red', linestyle=':', alpha=0.4, label='original k=8')
axes[1,1].set_title("Gap Statistic (higher = better)")
axes[1,1].set_xlabel("Number of clusters k")
axes[1,1].legend(fontsize=8)

plt.suptitle(f"Cluster Validation — {X.shape[0]} sigils, {n_components}-component PCA\n"
             f"Consensus recommendation: k={best_k}  (original: k=8)", fontsize=11)
plt.tight_layout()
plt.savefig(str(OUTDIR / "cluster_validation.png"), dpi=150, bbox_inches='tight')
print(f"\nSaved cluster_validation.png and cluster_validation.json")
