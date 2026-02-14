"""
Script 13: Template Matching - Recurring Sub-Motif Detection
Detects recurring visual sub-patterns (mini-templates) shared across multiple sigils.
Uses skeleton patch extraction + normalized cross-correlation to find shared motifs.
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")
SIGIL_DIR = OUTDIR / "extracted_sigils"
MOTIF_DIR = OUTDIR / "detected_motifs"
MOTIF_DIR.mkdir(exist_ok=True)

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

# ============================================================
# STEP 1: Extract candidate motif patches from each sigil
# ============================================================

def extract_motif_patches(binary, skel, patch_sizes=[24, 32, 40]):
    """Extract candidate motif patches centered on junction/endpoint regions."""
    padded = np.pad(skel, 1, mode='constant')
    nc = np.zeros_like(skel, dtype=int)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            nc += padded[1+dy:padded.shape[0]-1+dy, 1+dx:padded.shape[1]-1+dx]

    skel_mask = skel > 0

    # Interest points: junctions + endpoints
    interest = ((nc > 2) | (nc == 1)) & skel_mask
    ys, xs = np.where(interest)

    if len(ys) == 0:
        return []

    # Cluster nearby interest points and use centroids
    from scipy.ndimage import label as scipy_label
    dilated = cv2.dilate(interest.astype(np.uint8), np.ones((7,7), np.uint8))
    labeled, n_clusters = scipy_label(dilated)

    patches = []
    h, w = binary.shape

    for c in range(1, n_clusters + 1):
        cy_arr, cx_arr = np.where(labeled == c)
        cy, cx = int(np.mean(cy_arr)), int(np.mean(cx_arr))

        for ps in patch_sizes:
            half = ps // 2
            y1, y2 = cy - half, cy + half
            x1, x2 = cx - half, cx + half

            if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
                continue

            patch = binary[y1:y2, x1:x2]
            # Skip mostly empty patches
            if np.sum(patch > 0) / (ps * ps) < 0.03:
                continue

            patches.append({
                "patch": patch,
                "center": (cx, cy),
                "size": ps,
                "ink_ratio": float(np.sum(patch > 0) / (ps * ps))
            })

    return patches


# ============================================================
# STEP 2: Build patch library from all sigils
# ============================================================

print("=== Extracting Motif Patches from All Sigils ===")

all_patches = []  # list of (sigil_id, patch_info)
sigil_binaries = {}
sigil_skeletons = {}

for info in sigils[:72]:
    fpath = SIGIL_DIR / info["file"]
    if not fpath.exists():
        continue

    sid = info["id"]
    gray = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    skel = skeletonize(binary > 0).astype(np.uint8)

    sigil_binaries[sid] = binary
    sigil_skeletons[sid] = skel

    patches = extract_motif_patches(binary, skel, patch_sizes=[28, 36])
    for p in patches:
        all_patches.append((sid, p))

print(f"Extracted {len(all_patches)} candidate patches from {len(sigil_binaries)} sigils")


# ============================================================
# STEP 3: Use normalized cross-correlation to find shared motifs
# ============================================================

def match_patch_across_sigils(template_patch, sigil_binaries, source_id, threshold=0.7):
    """Find which sigils contain a region similar to template_patch."""
    matches = []
    template = template_patch.astype(np.float32)

    # Normalize template
    if template.std() == 0:
        return matches

    for sid, binary in sigil_binaries.items():
        if sid == source_id:
            continue

        target = binary.astype(np.float32)

        if target.shape[0] < template.shape[0] or target.shape[1] < template.shape[1]:
            continue

        try:
            result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val >= threshold:
                matches.append({
                    "sigil_id": sid,
                    "score": float(max_val),
                    "location": (int(max_loc[0]), int(max_loc[1]))
                })
        except:
            continue

    return matches


# Sample patches to test (testing all would be too slow)
# Pick up to 5 patches per sigil, prioritize high-ink patches
print("\n=== Searching for Shared Motifs (NCC threshold=0.65) ===")

patch_candidates = defaultdict(list)
for sid, p in all_patches:
    patch_candidates[sid].append(p)

# Select best patches per sigil
selected_patches = []
for sid in sorted(patch_candidates.keys()):
    patches = sorted(patch_candidates[sid], key=lambda p: p["ink_ratio"], reverse=True)
    for p in patches[:4]:  # top 4 per sigil
        selected_patches.append((sid, p))

print(f"Testing {len(selected_patches)} selected patches...")

# Track which motifs appear in multiple sigils
shared_motifs = []  # (source_id, patch_info, match_list)

for idx, (src_id, patch_info) in enumerate(selected_patches):
    if idx % 50 == 0:
        print(f"  Processing patch {idx}/{len(selected_patches)}...")

    matches = match_patch_across_sigils(
        patch_info["patch"], sigil_binaries, src_id, threshold=0.65
    )

    if len(matches) >= 2:  # found in at least 2 OTHER sigils
        shared_motifs.append({
            "source_id": src_id,
            "source_name": DEMON_NAMES.get(src_id, f"#{src_id}"),
            "patch_size": patch_info["size"],
            "patch_center": patch_info["center"],
            "ink_ratio": patch_info["ink_ratio"],
            "n_matches": len(matches),
            "matches": sorted(matches, key=lambda m: m["score"], reverse=True),
            "patch": patch_info["patch"]  # keep for visualization
        })

print(f"\nFound {len(shared_motifs)} motifs shared across 3+ sigils")

# Sort by number of matches (most shared first)
shared_motifs.sort(key=lambda m: m["n_matches"], reverse=True)


# ============================================================
# STEP 4: Deduplicate and categorize motifs
# ============================================================

# Motifs from nearby locations in same sigil are likely duplicates
# Keep only the most-shared version
deduped = []
seen_sources = set()

for motif in shared_motifs:
    key = (motif["source_id"], motif["patch_center"][0] // 15, motif["patch_center"][1] // 15)
    if key not in seen_sources:
        seen_sources.add(key)
        deduped.append(motif)

print(f"After deduplication: {len(deduped)} unique motifs")

# Take top 20
top_motifs = deduped[:20]


# ============================================================
# STEP 5: Visualize top shared motifs
# ============================================================

if top_motifs:
    n_show = min(10, len(top_motifs))
    fig, axes = plt.subplots(n_show, 6, figsize=(18, 3 * n_show))
    fig.suptitle("Top Shared Sub-Motifs Across Sigils", fontsize=16, fontweight='bold')

    if n_show == 1:
        axes = [axes]

    for row, motif in enumerate(top_motifs[:n_show]):
        # Column 0: the motif template
        axes[row][0].imshow(motif["patch"], cmap='gray_r')
        axes[row][0].set_title(f"Motif from #{motif['source_id']}\n{motif['source_name']}", fontsize=8)
        axes[row][0].axis('off')

        # Columns 1-5: best matches
        match_ids = [motif["source_id"]] + [m["sigil_id"] for m in motif["matches"][:5]]
        for col in range(1, 6):
            if col - 1 < len(motif["matches"]):
                match = motif["matches"][col - 1]
                msid = match["sigil_id"]
                if msid in sigil_binaries:
                    axes[row][col].imshow(sigil_binaries[msid], cmap='gray_r')
                    axes[row][col].set_title(
                        f"#{msid} {DEMON_NAMES.get(msid,'')}\nscore={match['score']:.2f}",
                        fontsize=7
                    )
                    # Draw rectangle at match location
                    loc = match["location"]
                    ps = motif["patch_size"]
                    rect = plt.Rectangle(loc, ps, ps, linewidth=2, edgecolor='red', facecolor='none')
                    axes[row][col].add_patch(rect)
                axes[row][col].axis('off')
            else:
                axes[row][col].axis('off')

    plt.tight_layout()
    plt.savefig(str(OUTDIR / "shared_motifs.png"), dpi=150)
    plt.close()

# ============================================================
# STEP 6: Build motif co-occurrence matrix
# ============================================================

# Which sigils share the most motifs?
cooccurrence = defaultdict(int)
sigil_motif_count = defaultdict(int)

for motif in deduped:
    involved = [motif["source_id"]] + [m["sigil_id"] for m in motif["matches"]]
    for i in range(len(involved)):
        sigil_motif_count[involved[i]] += 1
        for j in range(i + 1, len(involved)):
            pair = tuple(sorted([involved[i], involved[j]]))
            cooccurrence[pair] += 1

# Top co-occurring pairs
print("\n=== Top 15 Sigil Pairs Sharing Most Motifs ===")
sorted_pairs = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)
for (a, b), count in sorted_pairs[:15]:
    na = DEMON_NAMES.get(a, f"#{a}")
    nb = DEMON_NAMES.get(b, f"#{b}")
    print(f"  #{a} {na} <-> #{b} {nb}: {count} shared motifs")

# Sigils with most shared motifs
print("\n=== Sigils Participating in Most Shared Motifs ===")
sorted_sigils = sorted(sigil_motif_count.items(), key=lambda x: x[1], reverse=True)
for sid, count in sorted_sigils[:15]:
    print(f"  #{sid} {DEMON_NAMES.get(sid, '')}: participates in {count} shared motifs")

# Co-occurrence heatmap
if cooccurrence:
    all_ids = sorted(set([s for pair in cooccurrence for s in pair]))
    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    n = len(all_ids)
    matrix = np.zeros((n, n))
    for (a, b), count in cooccurrence.items():
        matrix[id_to_idx[a], id_to_idx[b]] = count
        matrix[id_to_idx[b], id_to_idx[a]] = count

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(s) for s in all_ids], fontsize=5, rotation=90)
    ax.set_yticklabels([str(s) for s in all_ids], fontsize=5)
    ax.set_title("Motif Co-occurrence Between Sigils")
    plt.colorbar(im, ax=ax, label="Number of shared motifs")
    plt.tight_layout()
    plt.savefig(str(OUTDIR / "motif_cooccurrence.png"), dpi=150)
    plt.close()

# ============================================================
# STEP 7: Save results
# ============================================================

# Save without the numpy patches
results_save = []
for motif in deduped:
    results_save.append({
        "source_id": motif["source_id"],
        "source_name": motif["source_name"],
        "patch_size": motif["patch_size"],
        "patch_center": list(motif["patch_center"]),
        "ink_ratio": round(motif["ink_ratio"], 4),
        "n_matches": motif["n_matches"],
        "matches": [{"sigil_id": m["sigil_id"], "score": round(m["score"], 4)}
                    for m in motif["matches"][:10]],
    })

output = {
    "n_total_motifs": len(deduped),
    "top_cooccurring_pairs": [
        {"pair": list(pair), "count": count,
         "names": [DEMON_NAMES.get(pair[0], ""), DEMON_NAMES.get(pair[1], "")]}
        for (pair, count) in sorted_pairs[:20]
    ],
    "most_connected_sigils": [
        {"id": sid, "name": DEMON_NAMES.get(sid, ""), "n_shared_motifs": count}
        for sid, count in sorted_sigils[:20]
    ],
    "motifs": results_save[:50],  # top 50
}

with open(OUTDIR / "template_matching.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved template_matching.json, shared_motifs.png, motif_cooccurrence.png")
