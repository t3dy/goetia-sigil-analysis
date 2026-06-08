"""
Script 25: Cross-Corpus Analysis

Projects precedent images (PGM karakteres, Solomonic seals, Ars Notoria notae, etc.)
into the same feature space as the 72 Goetic sigils, then measures:
  - Distance from each precedent to the nearest Goetia cluster centroid
  - Distance to the single most similar Goetia sigil
  - Which cluster each precedent is most similar to
  - Pairwise distances among precedents themselves

Inputs:
  - features.json (Goetia sigil features, from original pipeline or re-run)
  - cluster_assignments.json (from 06_clustering.py) — optional
  - docs/precedents/*.{jpg,png}  — sourced historical images

Outputs:
  - cross_corpus_features.json
  - cross_corpus_distances.json
  - cross_corpus_pca.png  (scatter: Goetia sigils + precedents overlaid)
"""

import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

OUTDIR     = Path(r"C:\Dev\GoetiaRevEng")
PREC_DIR   = OUTDIR / "docs" / "precedents"

# ── Feature extraction (same functions as 05_feature_extraction.py) ──────────

def box_counting_dimension(binary, min_box=2, max_box=64):
    sizes, counts = [], []
    s = min_box
    while s <= min(max_box, binary.shape[0], binary.shape[1]):
        count = sum(
            1 for y in range(0, binary.shape[0], s)
              for x in range(0, binary.shape[1], s)
              if np.any(binary[y:y+s, x:x+s] > 0)
        )
        if count > 0:
            sizes.append(s)
            counts.append(count)
        s *= 2
    if len(sizes) < 2:
        return 0
    coeffs = np.polyfit(np.log(1.0 / np.array(sizes)), np.log(np.array(counts)), 1)
    return round(coeffs[0], 3)


def symmetry_score(binary):
    b = binary.astype(float) / 255.0
    if b.size == 0 or np.std(b) == 0:
        return 0.0, 0.0
    h_sym = float(np.corrcoef(b.ravel(), np.fliplr(b).ravel())[0, 1])
    v_sym = float(np.corrcoef(b.ravel(), np.flipud(b).ravel())[0, 1])
    return round(h_sym, 3), round(v_sym, 3)


def radial_profile(binary, n_bins=8):
    h, w = binary.shape
    cy, cx = h / 2, w / 2
    max_r = np.sqrt(cx**2 + cy**2)
    ys, xs = np.mgrid[0:h, 0:w]
    dists_norm = np.sqrt((ys - cy)**2 + (xs - cx)**2) / max(max_r, 1)
    return [
        round(float(np.mean(binary[(dists_norm >= i/n_bins) & (dists_norm < (i+1)/n_bins)] > 0)), 4)
        if np.any((dists_norm >= i/n_bins) & (dists_norm < (i+1)/n_bins)) else 0
        for i in range(n_bins)
    ]


def quadrant_density(binary):
    h, w = binary.shape
    mh, mw = h // 2, w // 2
    return [round(float(np.mean(q > 0)), 4) for q in [
        binary[:mh, :mw], binary[:mh, mw:], binary[mh:, :mw], binary[mh:, mw:]
    ]]


def extract_features(fpath):
    """Load image, adaptive-binarize, extract the same 18-dim base feature vector."""
    img = cv2.imread(str(fpath))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # CLAHE + Otsu (matching 23_adaptive_preprocessing.py approach)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Resize to 256×256 for comparable feature computation
    binary = cv2.resize(binary, (256, 256), interpolation=cv2.INTER_AREA)
    binary = (binary > 127).astype(np.uint8) * 255

    ys, xs = np.where(binary > 0)
    ink_ratio   = float(np.mean(binary > 0))
    h, w = binary.shape
    aspect      = w / h
    compactness = (np.sum(binary > 0) / ((ys.max()-ys.min()+1)*(xs.max()-xs.min()+1))
                   if len(ys) > 0 else 0)
    fractal_d   = box_counting_dimension(binary)
    h_sym, v_sym = symmetry_score(binary)
    r_profile   = radial_profile(binary)
    q_density   = quadrant_density(binary)

    return {
        "ink_ratio":          round(ink_ratio, 4),
        "aspect_ratio":       round(aspect, 3),
        "compactness":        round(float(compactness), 3),
        "fractal_dimension":  fractal_d,
        "horizontal_symmetry": h_sym,
        "vertical_symmetry":  v_sym,
        "radial_profile":     r_profile,
        "quadrant_density":   q_density,
    }


def feat_to_vec(feat):
    """Convert feature dict to 18-dim vector (base features only, no skeleton/Hough)."""
    vec = [
        feat["ink_ratio"], feat["aspect_ratio"], feat["compactness"],
        feat["fractal_dimension"], feat["horizontal_symmetry"], feat["vertical_symmetry"],
    ]
    vec.extend(feat["radial_profile"])
    vec.extend(feat["quadrant_density"])
    # Pad to match 38-dim vectors from 06_clustering.py (zeros for skeleton/Hough)
    vec.extend([0] * 20)
    return vec


# ── Load Goetia features ──────────────────────────────────────────────────────

features_file = OUTDIR / "features.json"
if not features_file.exists():
    features_file = Path(r"C:\Users\PC\Downloads\goetia_analysis\features.json")

with open(features_file) as f:
    goetia_features = json.load(f)

cluster_file = OUTDIR / "cluster_assignments.json"
if not cluster_file.exists():
    cluster_file = Path(r"C:\Users\PC\Downloads\goetia_analysis\cluster_assignments.json")
cluster_map = {}  # sigil_id → cluster
if cluster_file.exists():
    with open(cluster_file) as f:
        ca = json.load(f)
    for cl_id, members in ca.items():
        for m in members:
            cluster_map[m] = int(cl_id)

goetia_vecs = []
goetia_ids  = []
for feat in goetia_features:
    goetia_vecs.append(feat_to_vec(feat))
    goetia_ids.append(feat["id"])

X_goetia = np.array(goetia_vecs)

# ── Extract features from precedent images ────────────────────────────────────

prec_images = sorted(PREC_DIR.glob("*.[jp][pn]g")) + sorted(PREC_DIR.glob("*.jpeg"))
print(f"Found {len(prec_images)} precedent images in {PREC_DIR}")

prec_records = []
prec_vecs    = []

for fpath in prec_images:
    feat = extract_features(fpath)
    if feat is None:
        print(f"  WARN: could not load {fpath.name}")
        continue

    label = fpath.stem.replace("_", " ").title()
    prec_records.append({"file": fpath.name, "label": label, **feat})
    prec_vecs.append(feat_to_vec(feat))
    print(f"  {fpath.name}: ink={feat['ink_ratio']:.3f}  fractal={feat['fractal_dimension']}")

if not prec_vecs:
    print("No precedent images found — check that docs/precedents/ has images.")
    exit(1)

X_prec = np.array(prec_vecs)

# ── Fit scaler + PCA on Goetia corpus, project precedents in ─────────────────

scaler = StandardScaler()
X_goetia_scaled = scaler.fit_transform(X_goetia)
X_prec_scaled   = scaler.transform(X_prec)   # same transform, not re-fit

n_comp = min(10, X_goetia_scaled.shape[0] - 1, X_goetia_scaled.shape[1])
pca = PCA(n_components=n_comp)
pca.fit(X_goetia_scaled)
X_goetia_pca = pca.transform(X_goetia_scaled)
X_prec_pca   = pca.transform(X_prec_scaled)

# ── Compute distances ─────────────────────────────────────────────────────────

# Each precedent → distance to every Goetia sigil (in PCA space)
dist_matrix = cdist(X_prec_pca, X_goetia_pca, metric='euclidean')

# Cluster centroids
goetia_clusters = [cluster_map.get(sid, 0) for sid in goetia_ids]
unique_clusters = sorted(set(goetia_clusters))
cluster_centroids = {}
for cl in unique_clusters:
    idx = [i for i, c in enumerate(goetia_clusters) if c == cl]
    cluster_centroids[cl] = X_goetia_pca[idx].mean(axis=0)

centroid_matrix = np.array([cluster_centroids[cl] for cl in unique_clusters])
prec_to_centroid = cdist(X_prec_pca, centroid_matrix, metric='euclidean')

distance_records = []
for pi, prec in enumerate(prec_records):
    nearest_sigil_idx = int(np.argmin(dist_matrix[pi]))
    nearest_sigil_id  = goetia_ids[nearest_sigil_idx]
    nearest_sigil_dist = float(dist_matrix[pi, nearest_sigil_idx])

    nearest_cluster_idx  = int(np.argmin(prec_to_centroid[pi]))
    nearest_cluster      = unique_clusters[nearest_cluster_idx]
    nearest_cluster_dist = float(prec_to_centroid[pi, nearest_cluster_idx])

    # All Goetia sigils within 1.5× nearest distance
    threshold  = nearest_sigil_dist * 1.5
    close_mask = dist_matrix[pi] <= threshold
    close_sigils = [goetia_ids[i] for i in np.where(close_mask)[0]]


    distance_records.append({
        "file":               prec["file"],
        "label":              prec["label"],
        "nearest_sigil":      nearest_sigil_id,
        "nearest_sigil_dist": round(nearest_sigil_dist, 3),
        "close_sigils":       close_sigils,
        "nearest_cluster":    nearest_cluster,
        "nearest_cluster_dist": round(nearest_cluster_dist, 3),
    })

    print(f"\n{prec['label']}:")
    print(f"  Nearest Goetia sigil: #{nearest_sigil_id} (dist={nearest_sigil_dist:.3f})")
    print(f"  Nearest cluster: {nearest_cluster} (dist={nearest_cluster_dist:.3f})")
    print(f"  Close sigils (<=1.5x): {close_sigils}")

# Pairwise distances among precedents
if len(prec_records) > 1:
    prec_pairwise = cdist(X_prec_pca, X_prec_pca)
    print("\n=== Most similar precedent pairs ===")
    pairs = []
    for i in range(len(prec_records)):
        for j in range(i+1, len(prec_records)):
            pairs.append((prec_records[i]["label"], prec_records[j]["label"],
                          float(prec_pairwise[i, j])))
    pairs.sort(key=lambda x: x[2])
    for a, b, d in pairs[:5]:
        print(f"  {a} <-> {b}: {d:.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────

with open(OUTDIR / "cross_corpus_features.json", "w") as f:
    json.dump(prec_records, f, indent=2)

with open(OUTDIR / "cross_corpus_distances.json", "w") as f:
    json.dump({
        "n_goetia_sigils": len(goetia_ids),
        "n_precedents":    len(prec_records),
        "pca_components":  n_comp,
        "pca_variance":    round(float(pca.explained_variance_ratio_.sum()), 4),
        "results":         distance_records,
    }, f, indent=2)

# ── PCA scatter plot ──────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 10))

# Goetia sigils, colored by cluster
goetia_colors = [goetia_clusters[i] for i in range(len(goetia_ids))]
scatter = ax.scatter(X_goetia_pca[:, 0], X_goetia_pca[:, 1],
                     c=goetia_colors, cmap='tab10', s=40, alpha=0.5,
                     edgecolors='none', label='Goetia sigils')
for i, sid in enumerate(goetia_ids):
    ax.annotate(str(sid), (X_goetia_pca[i, 0], X_goetia_pca[i, 1]),
                fontsize=5, color='#666666', ha='center', va='bottom')

# Precedent images — large markers
prec_colors = plt.cm.Set1(np.linspace(0, 1, len(prec_records)))
for pi, prec in enumerate(prec_records):
    ax.scatter(X_prec_pca[pi, 0], X_prec_pca[pi, 1],
               s=200, marker='*', color=prec_colors[pi],
               edgecolors='black', linewidth=0.8, zorder=5,
               label=prec["label"])
    ax.annotate(prec["label"], (X_prec_pca[pi, 0], X_prec_pca[pi, 1]),
                fontsize=7, fontweight='bold', ha='left', va='bottom',
                xytext=(4, 4), textcoords='offset points',
                color=prec_colors[pi])

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("Cross-Corpus PCA: Goetia Sigils + Precedents\n"
             "Stars = precedent images; circles = Goetia sigils (colored by cluster)")
ax.legend(loc='lower right', fontsize=7, framealpha=0.8)
plt.colorbar(scatter, ax=ax, label='Goetia cluster', shrink=0.6)
plt.tight_layout()
plt.savefig(str(OUTDIR / "cross_corpus_pca.png"), dpi=150, bbox_inches='tight')
print(f"\nSaved cross_corpus_pca.png, cross_corpus_features.json, cross_corpus_distances.json")
