"""
Script 26: Stroke Vocabulary & Terminal Decoration Analysis

The original pipeline detected junctions and endpoints but didn't classify
what those endpoints look like — the decorative elements (circles, triangles,
forked terminals, plain cuts) that are the most legible "vocabulary" shared
between Goetia sigils and their precedents.

This script:
  1. Skeletonizes each sigil with Zhang-Suen
  2. Locates all skeleton endpoints
  3. Extracts a small patch (21×21 px) around each endpoint
  4. Classifies terminal type using shape descriptors:
       - 'circle'   : high circularity, ring-like density profile
       - 'blob'     : compact filled region (dot, filled circle)
       - 'fork'     : 2+ branches near endpoint
       - 'plain'    : bare cut, no decoration
  5. Counts terminal types per sigil and outputs a vocabulary histogram

Inputs:  normalized_sigils/ (from 23) OR extracted_sigils_old/
Outputs: stroke_vocabulary.json, terminal_type_grid.png
"""

import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

OUTDIR    = Path(r"C:\Dev\GoetiaRevEng")
NORM_DIR  = OUTDIR / "normalized_sigils"
SIGIL_DIR = OUTDIR / "extracted_sigils_old"

# Use normalized sigils if available, otherwise fall back
input_dir = NORM_DIR if NORM_DIR.exists() and any(NORM_DIR.glob("*.png")) else SIGIL_DIR
print(f"Using sigils from: {input_dir}")

PATCH = 21   # px half-window around each endpoint
HALF  = PATCH // 2


def adaptive_binarize(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return binary


def get_skeleton_endpoints(skel_bool):
    """Return (row, col) of all skeleton endpoints (exactly 1 neighbor in 8-conn)."""
    kernel = np.ones((3, 3), np.uint8)
    neighbor_count = cv2.filter2D(skel_bool.astype(np.uint8), -1, kernel)
    # endpoint: skeleton pixel with exactly 2 total (itself + 1 neighbor)
    endpoint_mask = skel_bool & (neighbor_count == 2)
    ys, xs = np.where(endpoint_mask)
    return list(zip(ys.tolist(), xs.tolist()))


def classify_terminal(patch_binary):
    """
    Classify the decoration at a skeleton endpoint from a 21×21 binary patch.
    Returns one of: 'circle', 'blob', 'fork', 'plain'
    """
    if patch_binary.size == 0 or np.sum(patch_binary > 0) < 3:
        return 'plain'

    # How much ink is there?
    ink_ratio = np.mean(patch_binary > 0)

    # Detect contours for shape analysis
    contours, _ = cv2.findContours(patch_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 'plain'

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    # Circularity: 4π·area/perimeter² = 1 for perfect circle
    circularity = (4 * np.pi * area / perimeter**2) if perimeter > 0 else 0

    # Count ring structure: hollow vs filled
    # Flood-fill from center to measure enclosed area
    center_val = patch_binary[HALF, HALF]

    # Check for a ring: detect inner contours (holes)
    contours_all, hierarchy = cv2.findContours(
        patch_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    has_hole = (hierarchy is not None and
                hierarchy.shape[1] > 0 and
                np.any(hierarchy[0, :, 3] >= 0))   # parent exists → it's a hole

    # Fork detection: thin the patch and count branches at center
    skel_patch = skeletonize(patch_binary > 0)
    k = np.ones((3, 3), np.uint8)
    neighbor_count_p = cv2.filter2D(skel_patch.astype(np.uint8), -1, k)
    junction_pixels  = np.sum((skel_patch) & (neighbor_count_p >= 4))

    if junction_pixels >= 1:
        return 'fork'
    if circularity > 0.65 and has_hole and area > 20:
        return 'circle'
    if ink_ratio > 0.25 and area > 30:
        return 'blob'
    return 'plain'


# ── Main loop ──────────────────────────────────────────────────────────────────

sigil_files = sorted(input_dir.glob("*.png"))
vocabulary_records = []
example_patches = {'circle': [], 'blob': [], 'fork': [], 'plain': []}

for fpath in sigil_files:
    sid_str = fpath.stem
    try:
        sid = int(sid_str)
    except ValueError:
        sid = sid_str

    gray = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue

    binary = adaptive_binarize(gray)
    skel_bool = skeletonize(binary > 0)

    endpoints = get_skeleton_endpoints(skel_bool)
    h, w = binary.shape

    type_counts = {'circle': 0, 'blob': 0, 'fork': 0, 'plain': 0}
    for (ey, ex) in endpoints:
        y0 = max(0, ey - HALF)
        y1 = min(h, ey + HALF + 1)
        x0 = max(0, ex - HALF)
        x1 = min(w, ex + HALF + 1)
        patch = binary[y0:y1, x0:x1]
        if patch.shape[0] < 5 or patch.shape[1] < 5:
            continue

        # Pad to PATCH×PATCH for consistent analysis
        pad_y = PATCH - patch.shape[0]
        pad_x = PATCH - patch.shape[1]
        patch = np.pad(patch, ((0, pad_y), (0, pad_x)), constant_values=0)

        ttype = classify_terminal(patch)
        type_counts[ttype] += 1

        # Collect examples
        if len(example_patches[ttype]) < 8:
            example_patches[ttype].append((sid, patch.copy()))

    total = max(1, sum(type_counts.values()))
    vocabulary_records.append({
        "id":       sid,
        "n_endpoints": len(endpoints),
        "terminal_types": type_counts,
        "terminal_fractions": {
            k: round(v / total, 3) for k, v in type_counts.items()
        },
        "dominant_terminal": max(type_counts, key=type_counts.get),
    })

vocabulary_records.sort(key=lambda r: r["id"] if isinstance(r["id"], int) else 0)

with open(OUTDIR / "stroke_vocabulary.json", "w") as f:
    json.dump(vocabulary_records, f, indent=2)

print(f"Analyzed {len(vocabulary_records)} sigils")

# ── Summary ───────────────────────────────────────────────────────────────────

type_totals = {'circle': 0, 'blob': 0, 'fork': 0, 'plain': 0}
for r in vocabulary_records:
    for k, v in r["terminal_types"].items():
        type_totals[k] += v

total_terminals = sum(type_totals.values())
print("\n=== Terminal Type Distribution (corpus-wide) ===")
for ttype, count in sorted(type_totals.items(), key=lambda x: -x[1]):
    pct = 100 * count / max(1, total_terminals)
    print(f"  {ttype:8s}: {count:4d}  ({pct:.1f}%)")

dominant_counts = {}
for r in vocabulary_records:
    d = r["dominant_terminal"]
    dominant_counts[d] = dominant_counts.get(d, 0) + 1
print("\nDominant terminal per sigil:")
for ttype, n in sorted(dominant_counts.items(), key=lambda x: -x[1]):
    print(f"  {ttype:8s}: {n} sigils")

# ── Visualization: patch examples per type ────────────────────────────────────

fig, axes = plt.subplots(4, 8, figsize=(14, 8))
for row_i, ttype in enumerate(['circle', 'blob', 'fork', 'plain']):
    examples = example_patches[ttype]
    for col_i in range(8):
        ax = axes[row_i, col_i]
        ax.axis('off')
        if col_i < len(examples):
            sid, patch = examples[col_i]
            ax.imshow(patch, cmap='gray_r', vmin=0, vmax=255)
            ax.set_title(str(sid), fontsize=6)
        if col_i == 0:
            ax.set_ylabel(ttype, fontsize=8, labelpad=2)

plt.suptitle("Terminal Decoration Examples by Type\n(rows: circle / blob / fork / plain)",
             fontsize=10)
plt.tight_layout()
plt.savefig(str(OUTDIR / "terminal_type_grid.png"), dpi=150, bbox_inches='tight')
print("\nSaved stroke_vocabulary.json and terminal_type_grid.png")

# ── Bar chart by sigil ────────────────────────────────────────────────────────

ids_sorted  = [r["id"] for r in vocabulary_records if isinstance(r["id"], int)]
records_int = [r for r in vocabulary_records if isinstance(r["id"], int)]

circle_f = [r["terminal_fractions"]["circle"] for r in records_int]
blob_f   = [r["terminal_fractions"]["blob"]   for r in records_int]
fork_f   = [r["terminal_fractions"]["fork"]   for r in records_int]
plain_f  = [r["terminal_fractions"]["plain"]  for r in records_int]

x = np.arange(len(ids_sorted))
fig2, ax2 = plt.subplots(figsize=(18, 5))
ax2.bar(x, plain_f,  label='plain',  color='#aaaaaa')
ax2.bar(x, fork_f,   label='fork',   color='#e07040', bottom=plain_f)
bottom2 = [p + f for p, f in zip(plain_f, fork_f)]
ax2.bar(x, blob_f,   label='blob',   color='#4080c0', bottom=bottom2)
bottom3 = [b + bl for b, bl in zip(bottom2, blob_f)]
ax2.bar(x, circle_f, label='circle', color='#50c080', bottom=bottom3)

ax2.set_xticks(x)
ax2.set_xticklabels([str(i) for i in ids_sorted], fontsize=5, rotation=90)
ax2.set_ylabel("Fraction of terminals")
ax2.set_xlabel("Sigil number")
ax2.set_title("Terminal Decoration Vocabulary per Sigil")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig(str(OUTDIR / "terminal_vocabulary_bar.png"), dpi=150, bbox_inches='tight')
print("Saved terminal_vocabulary_bar.png")
