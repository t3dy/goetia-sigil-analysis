"""
Script 14: Historical Ordering Analysis
Examines whether sigil complexity or structural features correlate with their
ordering in the Goetia, to detect possible editorial patterns, scribal fatigue,
or systematic construction rules across the grimoire sequence.
"""
import cv2
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau, linregress
from scipy.signal import savgol_filter
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")

# Load all analysis data
with open(OUTDIR / "features.json") as f:
    features = json.load(f)
with open(OUTDIR / "skeleton_analysis.json") as f:
    skel_data = json.load(f)
with open(OUTDIR / "hough_analysis.json") as f:
    hough_data = json.load(f)
with open(OUTDIR / "junction_analysis.json") as f:
    junc_data = json.load(f)
with open(OUTDIR / "demon_metadata.json") as f:
    demon_meta = json.load(f)

feat_map = {f["id"]: f for f in features}
skel_map = {s["id"]: s for s in skel_data}
hough_map = {h["id"]: h for h in hough_data}
junc_map = {j["id"]: j for j in junc_data}
meta_map = {d["id"]: d for d in demon_meta}

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
# STEP 1: Compile sequential feature time series
# ============================================================

print("=== Historical Ordering Analysis ===")
print("Examining sigil features as a function of their ordering in the Goetia\n")

# Build ordered arrays
ids = list(range(1, 73))
ordered_features = {
    "fractal_dimension": [],
    "ink_ratio": [],
    "junctions": [],
    "endpoints": [],
    "holes": [],
    "n_lines": [],
    "n_circles": [],
    "horizontal_symmetry": [],
    "vertical_symmetry": [],
    "compactness": [],
    "aspect_ratio": [],
    "components": [],
}

for sid in ids:
    if sid in feat_map:
        ordered_features["fractal_dimension"].append(feat_map[sid]["fractal_dimension"])
        ordered_features["ink_ratio"].append(feat_map[sid]["ink_ratio"])
        ordered_features["horizontal_symmetry"].append(feat_map[sid]["horizontal_symmetry"])
        ordered_features["vertical_symmetry"].append(feat_map[sid]["vertical_symmetry"])
        ordered_features["compactness"].append(feat_map[sid]["compactness"])
        ordered_features["aspect_ratio"].append(feat_map[sid]["aspect_ratio"])
    else:
        for k in ["fractal_dimension","ink_ratio","horizontal_symmetry","vertical_symmetry","compactness","aspect_ratio"]:
            ordered_features[k].append(np.nan)

    if sid in skel_map:
        ordered_features["junctions"].append(skel_map[sid]["junctions"])
        ordered_features["endpoints"].append(skel_map[sid]["endpoints"])
        ordered_features["holes"].append(skel_map[sid]["holes"])
        ordered_features["components"].append(skel_map[sid]["connected_components"])
    else:
        for k in ["junctions","endpoints","holes","components"]:
            ordered_features[k].append(np.nan)

    if sid in hough_map:
        ordered_features["n_lines"].append(hough_map[sid]["n_lines"])
        ordered_features["n_circles"].append(hough_map[sid]["n_circles"])
    else:
        ordered_features["n_lines"].append(np.nan)
        ordered_features["n_circles"].append(np.nan)


# ============================================================
# STEP 2: Compute composite complexity score
# ============================================================

# Z-score normalize each feature and average for a composite complexity
from sklearn.preprocessing import StandardScaler

feature_names = list(ordered_features.keys())
matrix = np.array([ordered_features[k] for k in feature_names]).T  # (72, n_features)

# Handle NaN
valid_mask = ~np.isnan(matrix).any(axis=1)
valid_ids = [ids[i] for i in range(len(ids)) if valid_mask[i]]
valid_matrix = matrix[valid_mask]

scaler = StandardScaler()
normalized = scaler.fit_transform(valid_matrix)
complexity_score = np.mean(normalized, axis=1)

# Map back
complexity_map = {sid: float(score) for sid, score in zip(valid_ids, complexity_score)}

# ============================================================
# STEP 3: Statistical tests - ordering vs features
# ============================================================

print("=== Spearman Correlations: Sequence Position vs Feature ===")
results_corr = {}

for feat_name in feature_names:
    vals = ordered_features[feat_name]
    valid_pairs = [(i+1, v) for i, v in enumerate(vals) if not np.isnan(v)]
    if len(valid_pairs) < 10:
        continue

    positions, values = zip(*valid_pairs)
    rho, p = spearmanr(positions, values)
    tau, p_tau = kendalltau(positions, values)

    results_corr[feat_name] = {
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p), 4),
        "kendall_tau": round(float(tau), 4),
        "kendall_p": round(float(p_tau), 4),
        "significant": bool(p < 0.05)
    }

    sig = "*" if p < 0.05 else " "
    print(f"  {sig} {feat_name:25s}: rho={rho:+.3f}  p={p:.4f}  tau={tau:+.3f}")

# Composite complexity vs ordering
pos_valid = [i+1 for i in range(72) if valid_mask[i]]
rho_comp, p_comp = spearmanr(pos_valid, complexity_score)
print(f"\n  Composite complexity vs sequence: rho={rho_comp:+.3f}  p={p_comp:.4f}")


# ============================================================
# STEP 4: Detect trends and changepoints
# ============================================================

# Smoothed complexity curve using Savitzky-Golay filter
if len(complexity_score) >= 11:
    smoothed = savgol_filter(complexity_score, window_length=11, polyorder=2)
else:
    smoothed = complexity_score

# Split into thirds and compare
third = len(complexity_score) // 3
early = complexity_score[:third]
middle = complexity_score[third:2*third]
late = complexity_score[2*third:]

from scipy.stats import mannwhitneyu

try:
    u_em, p_em = mannwhitneyu(early, middle, alternative='two-sided')
    u_ml, p_ml = mannwhitneyu(middle, late, alternative='two-sided')
    u_el, p_el = mannwhitneyu(early, late, alternative='two-sided')
except:
    p_em = p_ml = p_el = 1.0

print(f"\n=== Complexity by Position (thirds) ===")
print(f"  Early  (1-24):  mean={np.mean(early):+.3f}, std={np.std(early):.3f}")
print(f"  Middle (25-48): mean={np.mean(middle):+.3f}, std={np.std(middle):.3f}")
print(f"  Late   (49-72): mean={np.mean(late):+.3f}, std={np.std(late):.3f}")
print(f"  Early vs Middle: p={p_em:.4f}")
print(f"  Middle vs Late:  p={p_ml:.4f}")
print(f"  Early vs Late:   p={p_el:.4f}")


# ============================================================
# STEP 5: Rank-based grouping analysis
# ============================================================

# Demons are ordered by rank (Kings first, then Dukes, etc.)
# Check if rank boundaries correspond to structural shifts

rank_groups = defaultdict(list)
for sid in valid_ids:
    if sid in meta_map:
        rank = meta_map[sid]["rank"]
        rank_groups[rank].append(complexity_map[sid])

print(f"\n=== Complexity by Rank ===")
for rank in ["King", "Duke", "Prince", "Marquis", "Earl/Count", "Knight", "President"]:
    if rank in rank_groups:
        vals = rank_groups[rank]
        print(f"  {rank:15s}: n={len(vals):2d}, mean={np.mean(vals):+.3f}, std={np.std(vals):.3f}")


# ============================================================
# STEP 6: Sliding window variance (scribal consistency)
# ============================================================

# High variance = inconsistent drawing; low = methodical
window = 8
variances = []
window_positions = []
for i in range(len(complexity_score) - window + 1):
    variances.append(np.var(complexity_score[i:i+window]))
    window_positions.append(i + window // 2 + 1)

print(f"\n=== Scribal Consistency (sliding window variance, w={window}) ===")
print(f"  Most consistent region: around sigil #{window_positions[np.argmin(variances)]}")
print(f"  Most variable region:   around sigil #{window_positions[np.argmax(variances)]}")
print(f"  Overall variance trend: {'increasing' if variances[-1] > variances[0] else 'decreasing'}")


# ============================================================
# STEP 7: Visualization
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("Historical Ordering Analysis: Sigil Sequence Patterns", fontsize=16, fontweight='bold')

# 1. Composite complexity over sequence
ax = axes[0, 0]
ax.scatter(pos_valid, complexity_score, s=25, alpha=0.6, c='steelblue', edgecolors='black', linewidth=0.5)
ax.plot(pos_valid, smoothed, 'r-', linewidth=2, alpha=0.8, label='Smoothed trend')
slope, intercept, r, p_lr, se = linregress(pos_valid, complexity_score)
x_line = np.array([1, 72])
ax.plot(x_line, slope * x_line + intercept, 'g--', linewidth=1.5,
        label=f'Linear fit (r={r:.3f}, p={p_lr:.3f})')
ax.set_xlabel("Goetia Sequence #")
ax.set_ylabel("Composite Complexity Score")
ax.set_title("Complexity vs Ordering Position")
ax.legend(fontsize=8)
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')

# 2. Key features over sequence
ax = axes[0, 1]
for feat, color, label in [
    ("fractal_dimension", "steelblue", "Fractal Dim"),
    ("junctions", "coral", "Junctions"),
    ("endpoints", "mediumseagreen", "Endpoints"),
]:
    vals = ordered_features[feat]
    valid = [(i+1, v) for i, v in enumerate(vals) if not np.isnan(v)]
    if valid:
        pos, v = zip(*valid)
        # Z-score for comparison
        v_arr = np.array(v)
        v_norm = (v_arr - v_arr.mean()) / (v_arr.std() + 1e-8)
        ax.plot(pos, v_norm, '.', alpha=0.4, color=color, markersize=4)
        if len(v_norm) >= 11:
            sm = savgol_filter(v_norm, 11, 2)
            ax.plot(pos, sm, '-', color=color, linewidth=2, label=label)
ax.set_xlabel("Goetia Sequence #")
ax.set_ylabel("Z-scored Feature Value")
ax.set_title("Feature Trends Across Sequence")
ax.legend(fontsize=8)

# 3. Sliding window variance
ax = axes[1, 0]
ax.fill_between(window_positions, variances, alpha=0.3, color='coral')
ax.plot(window_positions, variances, 'coral', linewidth=2)
ax.set_xlabel("Sequence Position (center of window)")
ax.set_ylabel("Local Variance")
ax.set_title(f"Scribal Consistency (window={window})")

# 4. Rank-colored complexity
ax = axes[1, 1]
rank_colors = {
    "King": "#E74C3C", "Duke": "#3498DB", "Prince": "#9B59B6",
    "Marquis": "#2ECC71", "Earl/Count": "#E67E22", "Knight": "#1ABC9C",
    "President": "#F39C12"
}
for sid in valid_ids:
    if sid in meta_map:
        rank = meta_map[sid]["rank"]
        color = rank_colors.get(rank, "#666")
        ax.scatter(sid, complexity_map[sid], c=color, s=40, edgecolors='black', linewidth=0.5)

# Legend
for rank, color in rank_colors.items():
    ax.scatter([], [], c=color, s=40, label=rank, edgecolors='black')
ax.legend(fontsize=7, ncol=2, loc='upper right')
ax.set_xlabel("Goetia Sequence #")
ax.set_ylabel("Complexity Score")
ax.set_title("Complexity Colored by Rank")

# 5. Autocorrelation of complexity
ax = axes[2, 0]
max_lag = 20
autocorrs = []
for lag in range(1, max_lag + 1):
    c1 = complexity_score[:-lag]
    c2 = complexity_score[lag:]
    r, _ = pearsonr(c1, c2)
    autocorrs.append(r)

ax.bar(range(1, max_lag + 1), autocorrs, color='steelblue', edgecolor='black', alpha=0.7)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axhline(2/np.sqrt(len(complexity_score)), color='red', linestyle='--', linewidth=1, alpha=0.5, label='95% CI')
ax.axhline(-2/np.sqrt(len(complexity_score)), color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")
ax.set_title("Complexity Autocorrelation (detect periodic patterns)")
ax.legend(fontsize=8)

# 6. Cumulative complexity (does it plateau?)
ax = axes[2, 1]
cumul = np.cumsum(complexity_score - np.mean(complexity_score))
ax.plot(pos_valid, cumul, 'steelblue', linewidth=2)
ax.fill_between(pos_valid, cumul, alpha=0.2, color='steelblue')
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlabel("Goetia Sequence #")
ax.set_ylabel("Cumulative Deviation from Mean")
ax.set_title("CUSUM Chart (detect regime changes)")

# Mark potential changepoints
if len(cumul) > 5:
    changepoint = pos_valid[np.argmax(np.abs(np.diff(cumul)))]
    ax.axvline(changepoint, color='red', linewidth=1.5, linestyle='--', alpha=0.7,
               label=f'Max shift ~#{changepoint}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(str(OUTDIR / "historical_ordering.png"), dpi=150)
plt.close()


# ============================================================
# STEP 8: Save results
# ============================================================

output = {
    "correlation_tests": results_corr,
    "composite_complexity": {
        "spearman_rho": round(float(rho_comp), 4),
        "spearman_p": round(float(p_comp), 4),
        "linear_fit": {
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 4),
            "r_value": round(float(r), 4),
            "p_value": round(float(p_lr), 4),
        }
    },
    "thirds_analysis": {
        "early_mean": round(float(np.mean(early)), 4),
        "middle_mean": round(float(np.mean(middle)), 4),
        "late_mean": round(float(np.mean(late)), 4),
        "early_vs_late_p": round(float(p_el), 4),
    },
    "rank_complexity": {
        rank: {
            "n": len(vals),
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
        }
        for rank, vals in rank_groups.items()
    },
    "scribal_consistency": {
        "most_consistent_position": int(window_positions[np.argmin(variances)]),
        "most_variable_position": int(window_positions[np.argmax(variances)]),
        "variance_trend": "increasing" if variances[-1] > variances[0] else "decreasing",
    },
    "complexity_scores": {str(sid): round(float(complexity_map[sid]), 4) for sid in valid_ids},
}

with open(OUTDIR / "historical_ordering.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved historical_ordering.json, historical_ordering.png")
