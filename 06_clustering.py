"""
Script 6: Hierarchical Clustering & Similarity Analysis
Groups sigils into families based on extracted features.
Also does motif/template matching to find shared sub-patterns.
"""
import cv2
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")
SIGIL_DIR = OUTDIR / "extracted_sigils"

# Load all analysis results
with open(OUTDIR / "features.json") as f:
    features = json.load(f)

skel_file = OUTDIR / "skeleton_analysis.json"
if skel_file.exists():
    with open(skel_file) as f:
        skel_data = json.load(f)
    skel_map = {s["id"]: s for s in skel_data}
else:
    skel_map = {}

hough_file = OUTDIR / "hough_analysis.json"
if hough_file.exists():
    with open(hough_file) as f:
        hough_data = json.load(f)
    hough_map = {h["id"]: h for h in hough_data}
else:
    hough_map = {}

# Build feature matrix
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

    # Add skeleton features if available
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

    # Add Hough features if available
    if sid in hough_map:
        h = hough_map[sid]
        vec.extend([
            h["n_lines"] / 100.0,
            h["n_circles"] / 20.0,
            h["line_circle_ratio"] / 50.0,
        ])
        # Angle histogram (normalized)
        ah = np.array(h["angle_histogram"], dtype=float)
        if ah.sum() > 0:
            ah = ah / ah.sum()
        vec.extend(ah.tolist())
    else:
        vec.extend([0, 0, 0] + [0]*12)

    feature_vectors.append(vec)
    sigil_ids.append(sid)

X = np.array(feature_vectors)
print(f"Feature matrix: {X.shape[0]} sigils × {X.shape[1]} features")

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hierarchical clustering
linkage_matrix = linkage(X_scaled, method='ward')

# Determine good number of clusters
n_clusters = 8  # Try 8 families

clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

# Dendrogram
fig, ax = plt.subplots(figsize=(20, 8))
dn = dendrogram(linkage_matrix, labels=[str(s) for s in sigil_ids],
                leaf_font_size=6, color_threshold=linkage_matrix[-n_clusters+1, 2],
                ax=ax)
ax.set_title(f"Hierarchical Clustering of 72 Goetic Sigils ({n_clusters} clusters)")
ax.set_xlabel("Sigil Number")
ax.set_ylabel("Ward Distance")
plt.tight_layout()
plt.savefig(str(OUTDIR / "dendrogram.png"), dpi=150)
print("Saved dendrogram.png")

# Print cluster assignments
cluster_groups = {}
for sid, cl in zip(sigil_ids, clusters):
    cluster_groups.setdefault(int(cl), []).append(sid)

print(f"\n=== {n_clusters} Sigil Families ===")
for cl in sorted(cluster_groups):
    members = cluster_groups[cl]
    print(f"\nCluster {cl} ({len(members)} sigils): {members}")

    # Average features for this cluster
    mask = np.array([sid in members for sid in sigil_ids])
    cluster_features = X[mask].mean(axis=0)

    # Find the most distinguishing features
    mean_all = X.mean(axis=0)
    std_all = X.std(axis=0)
    std_all[std_all == 0] = 1
    z_scores = (cluster_features - mean_all) / std_all

    feature_names = (
        ["ink_ratio", "aspect_ratio", "compactness", "fractal_dim",
         "h_symmetry", "v_symmetry"] +
        [f"radial_{i}" for i in range(8)] +
        [f"quad_{i}" for i in range(4)] +
        ["components", "junctions", "endpoints", "holes", "junc_end_ratio"] +
        ["n_lines", "n_circles", "line_circle_ratio"] +
        [f"angle_{i*15}" for i in range(12)]
    )

    top_features = np.argsort(np.abs(z_scores))[-3:][::-1]
    distinctions = []
    for idx in top_features:
        if idx < len(feature_names):
            name = feature_names[idx]
            direction = "high" if z_scores[idx] > 0 else "low"
            distinctions.append(f"{name}={direction}")
    print(f"  Distinctive: {', '.join(distinctions)}")

# PCA visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='tab10',
                     s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
for i, sid in enumerate(sigil_ids):
    ax.annotate(str(sid), (X_2d[i, 0], X_2d[i, 1]), fontsize=6,
                ha='center', va='bottom')
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("PCA of Goetic Sigil Features (colored by cluster)")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig(str(OUTDIR / "pca_clusters.png"), dpi=150)
print("\nSaved pca_clusters.png")

# Save cluster assignments
with open(OUTDIR / "cluster_assignments.json", "w") as f:
    json.dump({str(k): v for k, v in cluster_groups.items()}, f, indent=2)

# Find most similar pairs
from scipy.spatial.distance import squareform
dist_matrix = squareform(pdist(X_scaled))
np.fill_diagonal(dist_matrix, np.inf)

n_pairs = 10
flat_idx = np.argsort(dist_matrix.ravel())
print(f"\n=== Top {n_pairs} Most Similar Sigil Pairs ===")
seen = set()
count = 0
for idx in flat_idx:
    i, j = divmod(idx, len(sigil_ids))
    pair = (min(sigil_ids[i], sigil_ids[j]), max(sigil_ids[i], sigil_ids[j]))
    if pair not in seen:
        seen.add(pair)
        print(f"  Sigils {pair[0]} & {pair[1]}: distance = {dist_matrix[i,j]:.3f}")
        count += 1
        if count >= n_pairs:
            break
