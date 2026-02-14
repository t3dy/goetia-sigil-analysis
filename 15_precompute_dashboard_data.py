"""
Script 15: Precompute Dashboard Enhancement Data
Generates JSON files consumed by the interactive dashboard:
- fourier_neighbors.json: top 8 Fourier-similar sigils per demon
- spectral_neighbors.json: top 8 spectrally-similar sigils per demon
- feature_neighbors.json: top 8 feature-vector similar sigils per demon
- pca_coordinates.json: 2D PCA positions + loadings for interactive biplot
- cluster_feature_zscores.json: per-cluster z-score deviations
"""
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform, cosine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")

# Load existing analysis data
with open(OUTDIR / "fourier_descriptors.json") as f:
    fourier_data = json.load(f)
with open(OUTDIR / "graph_analysis.json") as f:
    graph_data = json.load(f)
with open(OUTDIR / "features.json") as f:
    features = json.load(f)
with open(OUTDIR / "skeleton_analysis.json") as f:
    skel_data = json.load(f)
with open(OUTDIR / "hough_analysis.json") as f:
    hough_data = json.load(f)
with open(OUTDIR / "cluster_assignments.json") as f:
    cluster_assignments = json.load(f)

# Build ID maps
feat_map = {f["id"]: f for f in features}
skel_map = {s["id"]: s for s in skel_data}
hough_map = {h["id"]: h for h in hough_data}

# ============================================================
# 1. Fourier similarity neighbors
# ============================================================
print("Computing Fourier neighbors...")
fourier_ids = [d["id"] for d in fourier_data]
fourier_matrix = np.array([d["fourier_descriptors"] for d in fourier_data])
fourier_dist = squareform(pdist(fourier_matrix, metric='cosine'))

fourier_neighbors = {}
for i, sid in enumerate(fourier_ids):
    dists = fourier_dist[i]
    dists_with_id = [(fourier_ids[j], float(dists[j])) for j in range(len(fourier_ids)) if j != i]
    dists_with_id.sort(key=lambda x: x[1])
    fourier_neighbors[str(sid)] = [{"id": nid, "distance": round(d, 4)} for nid, d in dists_with_id[:8]]

with open(OUTDIR / "docs" / "fourier_neighbors.json", "w") as f:
    json.dump(fourier_neighbors, f)
print(f"  Saved fourier_neighbors.json ({len(fourier_neighbors)} entries)")

# ============================================================
# 2. Spectral graph similarity neighbors
# ============================================================
print("Computing spectral neighbors...")
graph_ids = [d["id"] for d in graph_data]
spectral_matrix = []
for d in graph_data:
    fp = d.get("laplacian_eigenvalues", [])
    fp = fp + [0] * (10 - len(fp))
    spectral_matrix.append(fp[:10])
spectral_matrix = np.array(spectral_matrix)
spectral_dist = squareform(pdist(spectral_matrix, metric='euclidean'))

spectral_neighbors = {}
for i, sid in enumerate(graph_ids):
    dists = spectral_dist[i]
    dists_with_id = [(graph_ids[j], float(dists[j])) for j in range(len(graph_ids)) if j != i]
    dists_with_id.sort(key=lambda x: x[1])
    spectral_neighbors[str(sid)] = [{"id": nid, "distance": round(d, 4)} for nid, d in dists_with_id[:8]]

with open(OUTDIR / "docs" / "spectral_neighbors.json", "w") as f:
    json.dump(spectral_neighbors, f)
print(f"  Saved spectral_neighbors.json ({len(spectral_neighbors)} entries)")

# ============================================================
# 3. Feature-vector similarity neighbors
# ============================================================
print("Computing feature-vector neighbors...")
common_ids = sorted(set(feat_map.keys()) & set(skel_map.keys()) & set(hough_map.keys()))

feature_names = [
    "fractal_dimension", "ink_ratio", "compactness", "aspect_ratio",
    "horizontal_symmetry", "vertical_symmetry"
]
topo_names = ["junctions", "endpoints", "holes", "connected_components"]
geo_names = ["n_lines", "n_circles"]

rows = []
for sid in common_ids:
    row = []
    for fn in feature_names:
        row.append(feat_map[sid].get(fn, 0))
    for tn in topo_names:
        row.append(skel_map[sid].get(tn, 0))
    for gn in geo_names:
        row.append(hough_map[sid].get(gn, 0))
    rows.append(row)

all_feature_names = feature_names + topo_names + geo_names
feature_matrix = np.array(rows)
scaler = StandardScaler()
normalized = scaler.fit_transform(feature_matrix)
feature_dist = squareform(pdist(normalized, metric='euclidean'))

feature_neighbors = {}
for i, sid in enumerate(common_ids):
    dists = feature_dist[i]
    dists_with_id = [(common_ids[j], float(dists[j])) for j in range(len(common_ids)) if j != i]
    dists_with_id.sort(key=lambda x: x[1])
    feature_neighbors[str(sid)] = [{"id": nid, "distance": round(d, 4)} for nid, d in dists_with_id[:8]]

with open(OUTDIR / "docs" / "feature_neighbors.json", "w") as f:
    json.dump(feature_neighbors, f)
print(f"  Saved feature_neighbors.json ({len(feature_neighbors)} entries)")

# ============================================================
# 4. PCA coordinates + loadings for interactive biplot
# ============================================================
print("Computing PCA biplot data...")
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(normalized)

# Sigil-to-cluster mapping
sigil_to_cluster = {}
for cl, members in cluster_assignments.items():
    for m in members:
        sigil_to_cluster[m] = int(cl)

pca_data = {
    "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
    "feature_names": all_feature_names,
    "loadings": [[round(float(v), 4) for v in pca.components_[0]],
                 [round(float(v), 4) for v in pca.components_[1]]],
    "points": [
        {
            "id": int(common_ids[i]),
            "x": round(float(pca_coords[i, 0]), 4),
            "y": round(float(pca_coords[i, 1]), 4),
            "cluster": sigil_to_cluster.get(common_ids[i], 0)
        }
        for i in range(len(common_ids))
    ]
}

with open(OUTDIR / "docs" / "pca_biplot.json", "w") as f:
    json.dump(pca_data, f)
print(f"  Saved pca_biplot.json")

# ============================================================
# 5. Cluster feature z-scores
# ============================================================
print("Computing cluster z-scores...")
global_mean = np.mean(normalized, axis=0)
global_std = np.std(normalized, axis=0) + 1e-8

cluster_zscores = {}
for cl_str, members in cluster_assignments.items():
    cl_indices = [i for i, sid in enumerate(common_ids) if sid in members]
    if not cl_indices:
        continue
    cl_matrix = normalized[cl_indices]
    cl_mean = np.mean(cl_matrix, axis=0)
    zscores = (cl_mean - global_mean) / global_std
    sorted_features = sorted(zip(all_feature_names, zscores.tolist()), key=lambda x: abs(x[1]), reverse=True)
    cluster_zscores[cl_str] = [{"feature": fn, "zscore": round(z, 3)} for fn, z in sorted_features]

with open(OUTDIR / "docs" / "cluster_zscores.json", "w") as f:
    json.dump(cluster_zscores, f)
print(f"  Saved cluster_zscores.json")

print("\nAll dashboard enhancement data precomputed!")
