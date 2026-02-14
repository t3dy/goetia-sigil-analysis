"""
Script 12: Fourier Descriptors - Rotation-Invariant Contour Encoding
Encodes each sigil's outline as Fourier coefficients for rotation/scale-invariant
similarity comparison.
"""
import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")
SIGIL_DIR = OUTDIR / "extracted_sigils"

with open(OUTDIR / "sigil_metadata.json") as f:
    sigils = json.load(f)

DEMON_NAMES = {
    1:"Bael",2:"Agares",3:"Vassago",4:"Samigina",5:"Marbas",6:"Valefor",7:"Amon",
    8:"Barbatos",9:"Paimon",10:"Buer",11:"Gusion",12:"Sitri",13:"Beleth",14:"Leraje",
    15:"Eligos",16:"Zepar",17:"Botis",18:"Bathin",19:"Sallos",20:"Purson",21:"Marax",
    22:"Ipos",23:"Aim",24:"Naberius",25:"Glasya-Labolas",26:"Bune",27:"Ronove",
    28:"Berith",29:"Astaroth",30:"Forneus",31:"Foras",32:"Asmoday",33:"Gaap",
    34:"Furfur",35:"Marchosias",36:"Stolas",37:"Phenex",38:"Halphas",39:"Malphas",
    40:"Raum",41:"Focalor",42:"Vepar",43:"Sabnock",44:"Shax",45:"Vine",46:"Bifrons",
    47:"Uvall",48:"Haagenti",49:"Crocell",50:"Furcas",51:"Balam",52:"Alloces",
    53:"Camio",54:"Murmur",55:"Orobas",56:"Gremory",57:"Ose",58:"Amy",59:"Oriax",
    60:"Vapula",61:"Zagan",62:"Volac",63:"Andras",64:"Haures",65:"Andrealphus",
    66:"Cimejes",67:"Amdusias",68:"Belial",69:"Decarabia",70:"Seere",71:"Dantalion",
    72:"Andromalius"
}

def fourier_descriptors(contour, n_descriptors=32):
    """Compute Fourier descriptors from a contour."""
    if len(contour) < 4:
        return np.zeros(n_descriptors)

    # Flatten contour to complex representation
    contour = contour.squeeze()
    if contour.ndim != 2:
        return np.zeros(n_descriptors)

    z = contour[:, 0] + 1j * contour[:, 1]

    # DFT
    Z = np.fft.fft(z)

    # Normalize: make invariant to translation, scale, rotation, starting point
    # Remove DC component (translation invariance)
    Z[0] = 0

    # Scale invariance: divide by |Z[1]|
    if abs(Z[1]) > 0:
        Z = Z / abs(Z[1])

    # Rotation invariance: take magnitudes
    descriptors = np.abs(Z[1:n_descriptors+1])

    # Pad if needed
    if len(descriptors) < n_descriptors:
        descriptors = np.pad(descriptors, (0, n_descriptors - len(descriptors)))

    return descriptors


def multi_contour_descriptors(binary, n_descriptors=32, n_contours=5):
    """Get Fourier descriptors for the top-N largest contours, concatenated."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Sort by contour length
    contours = sorted(contours, key=lambda c: len(c), reverse=True)

    all_desc = []
    for cnt in contours[:n_contours]:
        fd = fourier_descriptors(cnt, n_descriptors)
        all_desc.append(fd)

    # Pad with zeros if fewer contours
    while len(all_desc) < n_contours:
        all_desc.append(np.zeros(n_descriptors))

    return np.concatenate(all_desc)


# Process all sigils
results = []
descriptors_matrix = []
sigil_ids = []

for info in sigils[:72]:
    fpath = SIGIL_DIR / info["file"]
    if not fpath.exists():
        continue

    sid = info["id"]
    gray = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # Get descriptors
    desc = multi_contour_descriptors(binary, n_descriptors=32, n_contours=5)

    results.append({
        "id": sid,
        "name": DEMON_NAMES.get(sid, f"#{sid}"),
        "fourier_descriptors": [round(float(d), 6) for d in desc],
        "n_descriptors": len(desc),
    })

    descriptors_matrix.append(desc)
    sigil_ids.append(sid)

# Compute distance matrix
desc_array = np.array(descriptors_matrix)
dist_matrix = squareform(pdist(desc_array, metric='cosine'))

# Find most Fourier-similar pairs
np.fill_diagonal(dist_matrix, np.inf)
print("=== Top 10 Most Fourier-Similar Pairs (Rotation-Invariant) ===")
seen = set()
flat_idx = np.argsort(dist_matrix.ravel())
count = 0
for idx in flat_idx:
    i, j = divmod(idx, len(sigil_ids))
    pair = (min(sigil_ids[i], sigil_ids[j]), max(sigil_ids[i], sigil_ids[j]))
    if pair not in seen:
        seen.add(pair)
        n1 = DEMON_NAMES.get(pair[0], f"#{pair[0]}")
        n2 = DEMON_NAMES.get(pair[1], f"#{pair[1]}")
        print(f"  #{pair[0]} {n1} <-> #{pair[1]} {n2}: cosine dist={dist_matrix[i,j]:.4f}")
        count += 1
        if count >= 10:
            break

# Fourier descriptor analysis
# Average descriptor magnitudes (which frequencies dominate?)
avg_desc_per_contour = np.mean(desc_array.reshape(len(desc_array), 5, 32), axis=0)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Average Fourier spectrum
axes[0,0].bar(range(32), avg_desc_per_contour[0], alpha=0.8, color='steelblue', edgecolor='black', linewidth=0.5)
axes[0,0].set_xlabel("Harmonic number")
axes[0,0].set_ylabel("Avg |F[k]|")
axes[0,0].set_title("Average Fourier Spectrum (Largest Contour)")

# 2. Distance heatmap (subset)
np.fill_diagonal(dist_matrix, 0)
im = axes[0,1].imshow(dist_matrix[:30, :30], cmap='viridis', aspect='auto')
axes[0,1].set_xticks(range(30))
axes[0,1].set_xticklabels([str(sigil_ids[i]) for i in range(30)], fontsize=5, rotation=90)
axes[0,1].set_yticks(range(30))
axes[0,1].set_yticklabels([str(sigil_ids[i]) for i in range(30)], fontsize=5)
axes[0,1].set_title("Fourier Distance (first 30 sigils)")
plt.colorbar(im, ax=axes[0,1])

# 3. PCA of Fourier descriptors
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
fd_2d = pca.fit_transform(desc_array)

with open(OUTDIR / "cluster_assignments.json") as f:
    clusters = json.load(f)
sigil_to_cluster = {}
for cl, members in clusters.items():
    for m in members:
        sigil_to_cluster[m] = int(cl)

CLUSTER_COLORS = {1:'#E67E22',2:'#2ECC71',3:'#3498DB',4:'#9B59B6',5:'#E74C3C',6:'#1ABC9C',7:'#F39C12',8:'#95A5A6'}
colors = [CLUSTER_COLORS.get(sigil_to_cluster.get(sid, 0), '#666') for sid in sigil_ids]

axes[1,0].scatter(fd_2d[:,0], fd_2d[:,1], c=colors, s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
for i, sid in enumerate(sigil_ids):
    axes[1,0].annotate(str(sid), (fd_2d[i,0], fd_2d[i,1]), fontsize=5, ha='center', va='bottom')
axes[1,0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[1,0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
axes[1,0].set_title("PCA of Fourier Descriptors (colored by cluster)")

# 4. Harmonic energy distribution
# How much of total energy is in low vs high harmonics?
for contour_idx in range(3):
    contour_descs = desc_array[:, contour_idx*32:(contour_idx+1)*32]
    energy = np.mean(contour_descs**2, axis=0)
    cumulative = np.cumsum(energy) / np.sum(energy) * 100
    axes[1,1].plot(range(32), cumulative, label=f'Contour {contour_idx+1}', linewidth=2)

axes[1,1].set_xlabel("Number of harmonics")
axes[1,1].set_ylabel("Cumulative energy (%)")
axes[1,1].set_title("How Many Harmonics Needed?")
axes[1,1].axhline(90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
axes[1,1].legend()

plt.suptitle("Fourier Descriptor Analysis", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(OUTDIR / "fourier_analysis.png"), dpi=150)
plt.close()

# Save results
with open(OUTDIR / "fourier_descriptors.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved fourier_descriptors.json, fourier_analysis.png")
print(f"Explained variance by first 2 PCs: {pca.explained_variance_ratio_[:2].sum():.1%}")
