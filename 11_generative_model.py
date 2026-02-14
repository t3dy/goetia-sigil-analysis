"""
Script 11: Generative Model - Probabilistic Sigil Grammar
Learns a construction grammar from the 72 sigils and generates new ones.
Uses the extracted features to build a probabilistic model of:
1. Spine orientation and length
2. Branch point distribution
3. Terminal decoration types
4. Enclosed loop probability
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")
SIGIL_DIR = OUTDIR / "extracted_sigils"
GEN_DIR = OUTDIR / "generated_sigils"
GEN_DIR.mkdir(exist_ok=True)

# Load analysis data to build the grammar
with open(OUTDIR / "features.json") as f:
    features = json.load(f)
with open(OUTDIR / "skeleton_analysis.json") as f:
    skel_data = json.load(f)
with open(OUTDIR / "hough_analysis.json") as f:
    hough_data = json.load(f)
with open(OUTDIR / "junction_analysis.json") as f:
    junc_data = json.load(f)

feat_map = {f["id"]: f for f in features}
skel_map = {s["id"]: s for s in skel_data}
hough_map = {h["id"]: h for h in hough_data}
junc_map = {j["id"]: j for j in junc_data}

# ============================================================
# LEARN THE GRAMMAR from the 72 sigils
# ============================================================

# 1. Aspect ratio distribution
aspects = [feat_map[i]["aspect_ratio"] for i in range(1,73) if i in feat_map]
aspect_mean, aspect_std = np.mean(aspects), np.std(aspects)

# 2. Number of main branches from center
junctions_list = [skel_map[i]["junctions"] for i in range(1,73) if i in skel_map]
endpoints_list = [skel_map[i]["endpoints"] for i in range(1,73) if i in skel_map]
components_list = [skel_map[i]["connected_components"] for i in range(1,73) if i in skel_map]
holes_list = [skel_map[i]["holes"] for i in range(1,73) if i in skel_map]

# 3. Angle distributions (aggregate all angle histograms)
all_angle_hists = []
for i in range(1,73):
    if i in hough_map:
        h = np.array(hough_map[i]["angle_histogram"], dtype=float)
        if h.sum() > 0:
            all_angle_hists.append(h / h.sum())
angle_probs = np.mean(all_angle_hists, axis=0) if all_angle_hists else np.ones(12)/12
angle_probs /= angle_probs.sum()

# 4. Terminal decoration probabilities
all_terminals = Counter()
for i in range(1,73):
    if i in junc_map:
        for tt, count in junc_map[i]["terminal_types"].items():
            all_terminals[tt] += count
total_terminals = sum(all_terminals.values())
terminal_probs = {k: v/total_terminals for k, v in all_terminals.items()}

# 5. Ink ratio distribution
ink_ratios = [feat_map[i]["ink_ratio"] for i in range(1,73) if i in feat_map]
ink_mean, ink_std = np.mean(ink_ratios), np.std(ink_ratios)

# 6. Symmetry distribution
h_syms = [feat_map[i]["horizontal_symmetry"] for i in range(1,73) if i in feat_map]
sym_prob = np.mean([s > 0.5 for s in h_syms])  # probability of being symmetric

# 7. Hole probability
hole_prob = np.mean([h > 0 for h in holes_list])

print("=== LEARNED GRAMMAR PARAMETERS ===")
print(f"Aspect ratio: mean={aspect_mean:.2f}, std={aspect_std:.2f}")
print(f"Junctions: mean={np.mean(junctions_list):.0f}, std={np.std(junctions_list):.0f}")
print(f"Endpoints: mean={np.mean(endpoints_list):.0f}, std={np.std(endpoints_list):.0f}")
print(f"Components: mean={np.mean(components_list):.0f}, std={np.std(components_list):.0f}")
print(f"Angle probabilities: {dict(zip([f'{i*15}-{(i+1)*15}' for i in range(12)], [f'{p:.3f}' for p in angle_probs]))}")
print(f"Terminal types: {terminal_probs}")
print(f"Symmetry probability: {sym_prob:.2f}")
print(f"Hole probability: {hole_prob:.2f}")

# ============================================================
# GENERATIVE ENGINE
# ============================================================

def draw_terminal(img, x, y, terminal_type, angle, size=4):
    """Draw a terminal decoration at a point."""
    if terminal_type == "circle":
        cv2.circle(img, (int(x), int(y)), size, 0, 2)
    elif terminal_type == "filled/cross":
        s = size
        cv2.line(img, (int(x-s), int(y)), (int(x+s), int(y)), 0, 2)
        cv2.line(img, (int(x), int(y-s)), (int(x), int(y+s)), 0, 2)
    elif terminal_type == "horizontal_bar":
        cv2.line(img, (int(x-size), int(y)), (int(x+size), int(y)), 0, 2)
    elif terminal_type == "vertical_bar":
        cv2.line(img, (int(x), int(y-size)), (int(x), int(y+size)), 0, 2)
    # "simple" = just the line endpoint, no decoration


def generate_sigil(canvas_size=200, seed=None):
    """Generate a new sigil using the learned grammar."""
    if seed is not None:
        np.random.seed(seed)

    img = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
    cx, cy = canvas_size // 2, canvas_size // 2

    # Decide if symmetric
    is_symmetric = np.random.random() < sym_prob

    # Number of main branches from center
    n_branches = np.random.randint(3, 8)

    # Sample angles from learned distribution
    angle_bins = np.random.choice(12, size=n_branches, p=angle_probs)
    angles = [bin_idx * 15 + np.random.uniform(0, 15) for bin_idx in angle_bins]

    if is_symmetric:
        # Mirror angles around vertical axis
        half = angles[:len(angles)//2+1]
        angles = half + [180 - a for a in half[:-1]]

    # Draw branches
    endpoints = []
    for angle_deg in angles:
        angle_rad = np.radians(angle_deg)
        length = np.random.randint(25, canvas_size//2 - 15)

        ex = cx + int(length * np.cos(angle_rad))
        ey = cy + int(length * np.sin(angle_rad))
        ex = np.clip(ex, 10, canvas_size-10)
        ey = np.clip(ey, 10, canvas_size-10)

        # Draw with slight jitter for organic feel
        cv2.line(img, (cx, cy), (ex, ey), 0, 1)
        endpoints.append((ex, ey, angle_deg))

        # Sub-branches with probability
        if np.random.random() < 0.5:
            n_sub = np.random.randint(1, 3)
            for _ in range(n_sub):
                # Branch point along the main branch
                t = np.random.uniform(0.3, 0.8)
                bx = int(cx + t * (ex - cx))
                by = int(cy + t * (ey - cy))

                sub_angle = angle_deg + np.random.choice([-90, -60, -45, 45, 60, 90])
                sub_len = np.random.randint(10, length//2)
                sub_rad = np.radians(sub_angle)

                sex = bx + int(sub_len * np.cos(sub_rad))
                sey = by + int(sub_len * np.sin(sub_rad))
                sex = np.clip(sex, 5, canvas_size-5)
                sey = np.clip(sey, 5, canvas_size-5)

                cv2.line(img, (bx, by), (sex, sey), 0, 1)
                endpoints.append((sex, sey, sub_angle))

    # Add enclosed loops with probability
    if np.random.random() < hole_prob:
        n_loops = np.random.randint(1, 3)
        for _ in range(n_loops):
            lx = cx + np.random.randint(-30, 30)
            ly = cy + np.random.randint(-30, 30)
            lw = np.random.randint(8, 25)
            lh = np.random.randint(8, 25)
            angle = np.random.randint(0, 180)
            cv2.ellipse(img, (lx, ly), (lw, lh), angle, 0, 360, 0, 1)

    # Add terminal decorations
    terminal_types = list(terminal_probs.keys())
    terminal_weights = list(terminal_probs.values())
    for ex, ey, angle in endpoints:
        tt = np.random.choice(terminal_types, p=terminal_weights)
        draw_terminal(img, ex, ey, tt, angle, size=np.random.randint(3, 6))

    # Add center decoration
    center_type = np.random.choice(["circle", "filled/cross", "simple"], p=[0.3, 0.4, 0.3])
    draw_terminal(img, cx, cy, center_type, 0, size=np.random.randint(3, 8))

    # Add horizontal/vertical bars (structural elements common in sigils)
    if np.random.random() < 0.6:
        n_bars = np.random.randint(1, 4)
        for _ in range(n_bars):
            bar_y = cy + np.random.randint(-40, 40)
            bar_x1 = cx - np.random.randint(15, 50)
            bar_x2 = cx + np.random.randint(15, 50)
            if np.random.random() < 0.5:
                cv2.line(img, (bar_x1, bar_y), (bar_x2, bar_y), 0, 1)
            else:
                bar_x = cx + np.random.randint(-40, 40)
                bar_y1 = cy - np.random.randint(15, 50)
                bar_y2 = cy + np.random.randint(15, 50)
                cv2.line(img, (bar_x, bar_y1), (bar_x, bar_y2), 0, 1)

    return img


# Generate 24 sigils
print("\n=== GENERATING 24 NEW SIGILS ===")
generated = []
for i in range(24):
    img = generate_sigil(200, seed=42+i)
    generated.append(img)
    cv2.imwrite(str(GEN_DIR / f"gen_{i+1:03d}.png"), img)

# Create a composite of generated sigils
fig, axes = plt.subplots(4, 6, figsize=(18, 12))
fig.suptitle("24 Generated Sigils (Probabilistic Grammar)", fontsize=16, fontweight='bold')
for i, img in enumerate(generated):
    ax = axes[i//6][i%6]
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(f"Gen #{i+1}", fontsize=9)
plt.tight_layout()
plt.savefig(str(OUTDIR / "generated_sigils_composite.png"), dpi=150)
plt.close()

# Compare generated vs real feature distributions
print("\nAnalyzing generated sigils...")
gen_fds = []
gen_endpoints_count = []
gen_junctions_count = []
gen_holes_count = []

for img in generated:
    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))

    skel = skeletonize(binary > 0).astype(np.uint8)
    padded = np.pad(skel, 1, mode='constant')
    nc = np.zeros_like(skel, dtype=int)
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            if dy==0 and dx==0: continue
            nc += padded[1+dy:padded.shape[0]-1+dy, 1+dx:padded.shape[1]-1+dx]

    skel_mask = skel > 0
    gen_junctions_count.append(int(np.sum((nc > 2) & skel_mask)))
    gen_endpoints_count.append(int(np.sum((nc == 1) & skel_mask)))

    # Holes
    contours_all, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_ext, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gen_holes_count.append(len(contours_all) - len(contours_ext))

    # Box-counting FD
    sizes = []
    counts = []
    s = 2
    while s <= 64:
        count = 0
        for y in range(0, binary.shape[0], s):
            for x in range(0, binary.shape[1], s):
                if np.any(binary[y:y+s, x:x+s] > 0):
                    count += 1
        if count > 0:
            sizes.append(s)
            counts.append(count)
        s *= 2
    if len(sizes) >= 2:
        coeffs = np.polyfit(np.log(1.0/np.array(sizes)), np.log(np.array(counts)), 1)
        gen_fds.append(coeffs[0])
    else:
        gen_fds.append(0)

# Comparison plot
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

axes[0].hist(gen_fds, bins=10, alpha=0.6, label='Generated', color='coral', edgecolor='black')
real_fds = [feat_map[i]["fractal_dimension"] for i in range(1,73) if i in feat_map]
axes[0].hist(real_fds, bins=10, alpha=0.6, label='Real', color='steelblue', edgecolor='black')
axes[0].set_xlabel("Fractal Dimension"); axes[0].set_title("FD: Real vs Generated"); axes[0].legend()

axes[1].hist(gen_junctions_count, bins=10, alpha=0.6, label='Generated', color='coral', edgecolor='black')
axes[1].hist(junctions_list, bins=10, alpha=0.6, label='Real', color='steelblue', edgecolor='black')
axes[1].set_xlabel("Junctions"); axes[1].set_title("Junctions: Real vs Generated"); axes[1].legend()

axes[2].hist(gen_endpoints_count, bins=10, alpha=0.6, label='Generated', color='coral', edgecolor='black')
axes[2].hist(endpoints_list, bins=10, alpha=0.6, label='Real', color='steelblue', edgecolor='black')
axes[2].set_xlabel("Endpoints"); axes[2].set_title("Endpoints: Real vs Generated"); axes[2].legend()

axes[3].hist(gen_holes_count, bins=10, alpha=0.6, label='Generated', color='coral', edgecolor='black')
axes[3].hist(holes_list, bins=10, alpha=0.6, label='Real', color='steelblue', edgecolor='black')
axes[3].set_xlabel("Holes"); axes[3].set_title("Holes: Real vs Generated"); axes[3].legend()

plt.suptitle("Feature Distribution Comparison: Real vs Generated Sigils", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(OUTDIR / "generated_vs_real_comparison.png"), dpi=150)
plt.close()

# Save grammar
grammar = {
    "aspect_ratio": {"mean": round(aspect_mean, 3), "std": round(aspect_std, 3)},
    "junctions": {"mean": round(float(np.mean(junctions_list)), 1), "std": round(float(np.std(junctions_list)), 1)},
    "endpoints": {"mean": round(float(np.mean(endpoints_list)), 1), "std": round(float(np.std(endpoints_list)), 1)},
    "angle_probabilities": {f"{i*15}-{(i+1)*15}": round(float(p), 4) for i, p in enumerate(angle_probs)},
    "terminal_probabilities": {k: round(v, 4) for k, v in terminal_probs.items()},
    "symmetry_probability": round(float(sym_prob), 3),
    "hole_probability": round(float(hole_prob), 3),
    "ink_ratio": {"mean": round(ink_mean, 4), "std": round(ink_std, 4)},
}
with open(OUTDIR / "learned_grammar.json", "w") as f:
    json.dump(grammar, f, indent=2)

print("\nSaved learned_grammar.json, generated_sigils_composite.png, generated_vs_real_comparison.png")
print(f"Generated {len(generated)} sigils in {GEN_DIR}")
