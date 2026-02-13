"""
Script 7: Cluster Composite Visualizations
Creates a labeled image grid for each sigil family, with feature summaries.
"""
import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SIGIL_DIR = Path(r"C:\Users\PC\Downloads\goetia_analysis\extracted_sigils")
OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")

# Goetia demon names (1-indexed, standard ordering)
DEMON_NAMES = {
    1: "Bael", 2: "Agares", 3: "Vassago", 4: "Samigina", 5: "Marbas",
    6: "Valefor", 7: "Amon", 8: "Barbatos", 9: "Paimon", 10: "Buer",
    11: "Gusion", 12: "Sitri", 13: "Beleth", 14: "Leraje", 15: "Eligos",
    16: "Zepar", 17: "Botis", 18: "Bathin", 19: "Sallos", 20: "Purson",
    21: "Marax", 22: "Ipos", 23: "Aim", 24: "Naberius", 25: "Glasya-Labolas",
    26: "Bune", 27: "Ronove", 28: "Berith", 29: "Astaroth", 30: "Forneus",
    31: "Foras", 32: "Asmoday", 33: "Gaap", 34: "Furfur", 35: "Marchosias",
    36: "Stolas", 37: "Phenex", 38: "Halphas", 39: "Malphas", 40: "Raum",
    41: "Focalor", 42: "Vepar", 43: "Sabnock", 44: "Shax", 45: "Vine",
    46: "Bifrons", 47: "Uvall", 48: "Haagenti", 49: "Crocell", 50: "Furcas",
    51: "Balam", 52: "Alloces", 53: "Camio", 54: "Murmur", 55: "Orobas",
    56: "Gremory", 57: "Ose", 58: "Amy", 59: "Oriax", 60: "Vapula",
    61: "Zagan", 62: "Volac", 63: "Andras", 64: "Haures", 65: "Andrealphus",
    66: "Cimejes", 67: "Amdusias", 68: "Belial", 69: "Decarabia", 70: "Seere",
    71: "Dantalion", 72: "Andromalius"
}

# Cluster descriptions based on analysis
CLUSTER_DESCRIPTIONS = {
    "1": {
        "name": "The Sparse Minimalists",
        "desc": "Low complexity, low ink density, low compactness.\nThese are the most skeletal and sparse sigils — minimal\nstrokes, open layouts, low fractal dimension.\nFew junctions, few holes.",
        "color": "#E67E22"
    },
    "2": {
        "name": "The Spread Networks",
        "desc": "Low ink ratio, low compactness, 75° angle preference.\nThe largest family — spread-out designs with moderate\nbranching. Strokes fan outward from center.\nBalanced between lines and curves.",
        "color": "#2ECC71"
    },
    "3": {
        "name": "The Circuit Loops",
        "desc": "High holes, high junction/endpoint ratio, high ink density.\nThe most 'circuit-like' — dense, enclosed loops,\nheavy branching networks. Highest fractal dimension.\nThese truly resemble wiring diagrams.",
        "color": "#3498DB"
    },
    "4": {
        "name": "The Diagonal Stars",
        "desc": "Strong 45° and 120° angle emphasis.\nOnly 2 members — both feature prominent diagonal\ncross-strokes with high horizontal symmetry.\nStar-like or X-shaped central motifs.",
        "color": "#9B59B6"
    },
    "5": {
        "name": "The Mixed-Angle Composites",
        "desc": "150° and 15° angle preferences, mixed orientation.\nModerately dense, these sigils avoid the strict\northogonal grid and use oblique angles.\nOften vertically symmetric.",
        "color": "#E74C3C"
    },
    "6": {
        "name": "The Wide Horizontals",
        "desc": "Wide aspect ratio, high line dominance, 0° angle emphasis.\nThese stretch horizontally — long lateral bars\nwith elements strung along them.\nStrong horizontal axis.",
        "color": "#1ABC9C"
    },
    "7": {
        "name": "The Unified Verticals",
        "desc": "Low component count, strong 90° vertical emphasis.\nMore connected designs with fewer separate pieces.\nVertical central spines with elements hanging off them.\nCompact radial profiles.",
        "color": "#F39C12"
    },
    "8": {
        "name": "The Outlier (Zepar)",
        "desc": "Extreme aspect ratio (3:1), edge-heavy radial profile.\nA unique structural outlier — the widest sigil\nwith ink concentrated at the periphery\nrather than the center.",
        "color": "#95A5A6"
    },
}

# Load data
with open(OUTDIR / "cluster_assignments.json") as f:
    clusters = json.load(f)

with open(OUTDIR / "features.json") as f:
    features = json.load(f)
feat_map = {f["id"]: f for f in features}

with open(OUTDIR / "skeleton_analysis.json") as f:
    skel_data = json.load(f)
skel_map = {s["id"]: s for s in skel_data}


def load_sigil_image(sigil_id, target_size=150):
    """Load and resize a sigil to a square thumbnail."""
    # Our segmented IDs don't perfectly match Goetia numbering due to
    # variant entries, so we use the segmented ID directly
    fname = f"sigil_{sigil_id:03d}.png"
    fpath = SIGIL_DIR / fname
    if not fpath.exists():
        # Return blank
        return np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    img = cv2.imread(str(fpath))
    h, w = img.shape[:2]

    # Pad to square
    max_dim = max(h, w)
    pad_img = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255
    y_off = (max_dim - h) // 2
    x_off = (max_dim - w) // 2
    pad_img[y_off:y_off+h, x_off:x_off+w] = img

    # Resize
    resized = cv2.resize(pad_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


def make_cluster_composite(cluster_id, sigil_ids, desc_info):
    """Create a composite image for one cluster."""
    n = len(sigil_ids)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(max(14, cols * 2.5), rows * 3 + 3.5))

    # Title area
    fig.suptitle(f"Cluster {cluster_id}: {desc_info['name']}\n({n} sigils)",
                 fontsize=18, fontweight='bold', color=desc_info['color'], y=0.98)

    gs = GridSpec(rows + 1, cols, figure=fig, hspace=0.4, wspace=0.3,
                  top=0.88, bottom=0.08, left=0.05, right=0.95)

    # Description text at top
    desc_ax = fig.add_subplot(gs[0, :])
    desc_ax.axis('off')

    # Build stats summary
    fds = [feat_map[s]["fractal_dimension"] for s in sigil_ids if s in feat_map]
    inks = [feat_map[s]["ink_ratio"] for s in sigil_ids if s in feat_map]
    h_syms = [feat_map[s]["horizontal_symmetry"] for s in sigil_ids if s in feat_map]
    v_syms = [feat_map[s]["vertical_symmetry"] for s in sigil_ids if s in feat_map]
    juncs = [skel_map[s]["junctions"] for s in sigil_ids if s in skel_map]
    ends = [skel_map[s]["endpoints"] for s in sigil_ids if s in skel_map]
    holes = [skel_map[s]["holes"] for s in sigil_ids if s in skel_map]
    comps = [skel_map[s]["connected_components"] for s in sigil_ids if s in skel_map]

    stats_text = (
        f"Avg fractal dim: {np.mean(fds):.2f}  |  "
        f"Avg ink ratio: {np.mean(inks):.3f}  |  "
        f"Avg symmetry (H/V): {np.mean(h_syms):.2f}/{np.mean(v_syms):.2f}  |  "
        f"Avg junctions: {np.mean(juncs):.0f}  |  "
        f"Avg endpoints: {np.mean(ends):.0f}  |  "
        f"Avg holes: {np.mean(holes):.1f}  |  "
        f"Avg components: {np.mean(comps):.0f}"
    )

    desc_ax.text(0.0, 0.7, desc_info['desc'], fontsize=10, fontfamily='monospace',
                 verticalalignment='top', transform=desc_ax.transAxes,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=desc_info['color'],
                           alpha=0.15, edgecolor=desc_info['color']))
    desc_ax.text(0.0, -0.1, stats_text, fontsize=8, fontfamily='monospace',
                 verticalalignment='top', transform=desc_ax.transAxes, color='#555')

    # Sigil images
    for i, sid in enumerate(sorted(sigil_ids)):
        row = (i // cols) + 1
        col = i % cols
        ax = fig.add_subplot(gs[row, col])

        img_rgb = cv2.cvtColor(load_sigil_image(sid), cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.axis('off')

        name = DEMON_NAMES.get(sid, f"#{sid}")
        fd = feat_map.get(sid, {}).get("fractal_dimension", 0)
        ax.set_title(f"#{sid} {name}\nFD={fd:.2f}", fontsize=8, pad=2)

    outpath = OUTDIR / f"cluster_{cluster_id}_composite.png"
    plt.savefig(str(outpath), dpi=150, facecolor='white')
    plt.close()
    print(f"Saved {outpath.name}")


# Generate composites for each cluster
for cl_id in sorted(clusters.keys(), key=int):
    sigil_ids = clusters[cl_id]
    desc = CLUSTER_DESCRIPTIONS.get(cl_id, {
        "name": f"Cluster {cl_id}",
        "desc": "Uncharacterized cluster.",
        "color": "#333"
    })
    make_cluster_composite(cl_id, sigil_ids, desc)


# Also make an overview: one representative from each cluster in a single image
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle("8 Sigil Families — Representative Examples", fontsize=16, fontweight='bold')

for idx, cl_id in enumerate(sorted(clusters.keys(), key=int)):
    ax = axes[idx // 4][idx % 4]
    members = clusters[cl_id]
    # Pick the member closest to cluster centroid
    member_feats = []
    for sid in members:
        if sid in feat_map:
            f = feat_map[sid]
            vec = [f["ink_ratio"], f["fractal_dimension"],
                   f["horizontal_symmetry"], f["vertical_symmetry"]]
            member_feats.append((sid, vec))

    if member_feats:
        vecs = np.array([v for _, v in member_feats])
        centroid = vecs.mean(axis=0)
        dists = np.linalg.norm(vecs - centroid, axis=1)
        rep_sid = member_feats[np.argmin(dists)][0]
    else:
        rep_sid = members[0]

    img_rgb = cv2.cvtColor(load_sigil_image(rep_sid, 200), cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.axis('off')

    desc = CLUSTER_DESCRIPTIONS.get(cl_id, {"name": f"Cluster {cl_id}", "color": "#333"})
    name = DEMON_NAMES.get(rep_sid, f"#{rep_sid}")
    ax.set_title(f"Cluster {cl_id}: {desc['name']}\n(rep: #{rep_sid} {name}, n={len(members)})",
                 fontsize=9, color=desc['color'], fontweight='bold')

plt.tight_layout()
plt.savefig(str(OUTDIR / "cluster_overview.png"), dpi=150, facecolor='white')
plt.close()
print("Saved cluster_overview.png")
