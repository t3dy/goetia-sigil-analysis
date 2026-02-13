"""
Script 5: Topological & Geometric Feature Extraction
Extracts a comprehensive feature vector per sigil for clustering:
- Fractal dimension (box-counting)
- Symmetry scores
- Stroke density distribution
- Radial profile
- Aspect ratio & compactness
"""
import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIGIL_DIR = Path(r"C:\Users\PC\Downloads\goetia_analysis\extracted_sigils")
OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")

metadata_file = OUTDIR / "sigil_metadata.json"
with open(metadata_file) as f:
    sigils = json.load(f)


def box_counting_dimension(binary, min_box=2, max_box=64):
    """Estimate fractal dimension via box counting."""
    sizes = []
    counts = []
    s = min_box
    while s <= min(max_box, binary.shape[0], binary.shape[1]):
        count = 0
        for y in range(0, binary.shape[0], s):
            for x in range(0, binary.shape[1], s):
                box = binary[y:y+s, x:x+s]
                if np.any(box > 0):
                    count += 1
        if count > 0:
            sizes.append(s)
            counts.append(count)
        s *= 2

    if len(sizes) < 2:
        return 0
    log_sizes = np.log(1.0 / np.array(sizes))
    log_counts = np.log(np.array(counts))
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return round(coeffs[0], 3)


def symmetry_score(binary):
    """Measure bilateral symmetry (horizontal and vertical flip correlation)."""
    h, w = binary.shape
    b = binary.astype(float) / 255.0

    # Horizontal symmetry (left-right)
    flipped_h = np.fliplr(b)
    if b.size > 0 and np.std(b) > 0:
        h_sym = np.corrcoef(b.ravel(), flipped_h.ravel())[0, 1]
    else:
        h_sym = 0

    # Vertical symmetry (top-bottom)
    flipped_v = np.flipud(b)
    if b.size > 0 and np.std(b) > 0:
        v_sym = np.corrcoef(b.ravel(), flipped_v.ravel())[0, 1]
    else:
        v_sym = 0

    return round(float(h_sym), 3), round(float(v_sym), 3)


def radial_profile(binary, n_bins=8):
    """Measure ink density at different radii from centroid."""
    h, w = binary.shape
    cy, cx = h / 2, w / 2
    max_r = np.sqrt(cx**2 + cy**2)

    ys, xs = np.mgrid[0:h, 0:w]
    dists = np.sqrt((ys - cy)**2 + (xs - cx)**2)
    dists_norm = dists / max_r  # 0 to 1

    profile = []
    for i in range(n_bins):
        r_lo = i / n_bins
        r_hi = (i + 1) / n_bins
        mask = (dists_norm >= r_lo) & (dists_norm < r_hi)
        if np.sum(mask) > 0:
            density = np.mean(binary[mask] > 0)
        else:
            density = 0
        profile.append(round(float(density), 4))
    return profile


def quadrant_density(binary):
    """Ink density in each quadrant."""
    h, w = binary.shape
    mh, mw = h // 2, w // 2
    quads = [
        binary[:mh, :mw],  # top-left
        binary[:mh, mw:],  # top-right
        binary[mh:, :mw],  # bottom-left
        binary[mh:, mw:],  # bottom-right
    ]
    return [round(float(np.mean(q > 0)), 4) for q in quads]


features = []
for info in sigils[:72]:
    fpath = SIGIL_DIR / info["file"]
    if not fpath.exists():
        continue

    gray = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # Basic metrics
    ink_ratio = float(np.mean(binary > 0))
    h, w = binary.shape
    aspect = w / h

    # Compactness: ratio of ink area to bounding box area of ink
    ys, xs = np.where(binary > 0)
    if len(ys) > 0:
        ink_bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        compactness = np.sum(binary > 0) / ink_bbox_area
    else:
        compactness = 0

    fractal_d = box_counting_dimension(binary)
    h_sym, v_sym = symmetry_score(binary)
    r_profile = radial_profile(binary)
    q_density = quadrant_density(binary)

    feat = {
        "id": info["id"],
        "ink_ratio": round(ink_ratio, 4),
        "aspect_ratio": round(aspect, 3),
        "compactness": round(float(compactness), 3),
        "fractal_dimension": fractal_d,
        "horizontal_symmetry": h_sym,
        "vertical_symmetry": v_sym,
        "radial_profile": r_profile,
        "quadrant_density": q_density,
    }
    features.append(feat)

with open(OUTDIR / "features.json", "w") as f:
    json.dump(features, f, indent=2)

print(f"Extracted features for {len(features)} sigils")

# Visualization: feature distributions
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Fractal dimension distribution
fds = [f["fractal_dimension"] for f in features]
axes[0,0].hist(fds, bins=20, color='coral', edgecolor='black')
axes[0,0].set_title("Fractal Dimension Distribution")
axes[0,0].set_xlabel("Box-counting dimension")

# Symmetry scatter
h_syms = [f["horizontal_symmetry"] for f in features]
v_syms = [f["vertical_symmetry"] for f in features]
axes[0,1].scatter(h_syms, v_syms, alpha=0.6, s=30)
axes[0,1].set_xlabel("Horizontal Symmetry")
axes[0,1].set_ylabel("Vertical Symmetry")
axes[0,1].set_title("Symmetry Scores")
axes[0,1].axhline(0, color='gray', linestyle='--', alpha=0.3)
axes[0,1].axvline(0, color='gray', linestyle='--', alpha=0.3)

# Ink ratio
inks = [f["ink_ratio"] for f in features]
axes[0,2].hist(inks, bins=20, color='steelblue', edgecolor='black')
axes[0,2].set_title("Ink Coverage Ratio")

# Average radial profile
all_profiles = np.array([f["radial_profile"] for f in features])
mean_profile = np.mean(all_profiles, axis=0)
std_profile = np.std(all_profiles, axis=0)
x = np.arange(len(mean_profile))
axes[1,0].bar(x, mean_profile, yerr=std_profile, color='mediumseagreen', alpha=0.8, capsize=3)
axes[1,0].set_xlabel("Radial bin (center → edge)")
axes[1,0].set_ylabel("Ink density")
axes[1,0].set_title("Average Radial Profile")

# Compactness vs fractal dimension
comps = [f["compactness"] for f in features]
axes[1,1].scatter(fds, comps, alpha=0.6, s=30, c='purple')
axes[1,1].set_xlabel("Fractal Dimension")
axes[1,1].set_ylabel("Compactness")
axes[1,1].set_title("Complexity vs Compactness")

# Quadrant balance (variance across quadrants)
q_vars = [np.var(f["quadrant_density"]) for f in features]
axes[1,2].hist(q_vars, bins=20, color='goldenrod', edgecolor='black')
axes[1,2].set_title("Quadrant Density Variance\n(low = balanced, high = asymmetric)")

plt.tight_layout()
plt.savefig(str(OUTDIR / "feature_distributions.png"), dpi=150)
print("Saved feature_distributions.png")

# Print most/least symmetric
sorted_sym = sorted(features, key=lambda f: f["horizontal_symmetry"] + f["vertical_symmetry"], reverse=True)
print("\nMost symmetric sigils:")
for f in sorted_sym[:5]:
    print(f"  Sigil {f['id']}: h_sym={f['horizontal_symmetry']}, v_sym={f['vertical_symmetry']}")
print("Least symmetric sigils:")
for f in sorted_sym[-5:]:
    print(f"  Sigil {f['id']}: h_sym={f['horizontal_symmetry']}, v_sym={f['vertical_symmetry']}")
