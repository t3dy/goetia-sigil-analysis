"""
Script 8: Textual-Structural Correlation Analysis
Correlates demon metadata (rank, legions, abilities, appearance) with
sigil structural features to find relationships between text and image.
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")

# Load everything
with open(OUTDIR / "demon_metadata.json") as f:
    demons = json.load(f)
demon_map = {d["id"]: d for d in demons}

with open(OUTDIR / "features.json") as f:
    features = json.load(f)
feat_map = {f["id"]: f for f in features}

with open(OUTDIR / "skeleton_analysis.json") as f:
    skel_data = json.load(f)
skel_map = {s["id"]: s for s in skel_data}

with open(OUTDIR / "hough_analysis.json") as f:
    hough_data = json.load(f)
hough_map = {h["id"]: h for h in hough_data}

with open(OUTDIR / "cluster_assignments.json") as f:
    clusters = json.load(f)

# Invert cluster map: sigil_id -> cluster
sigil_to_cluster = {}
for cl, members in clusters.items():
    for m in members:
        sigil_to_cluster[m] = int(cl)

# ============================================================
# ANALYSIS 1: Rank vs Structural Features
# ============================================================
RANKS = ["King", "Duke", "Prince", "Marquis", "Earl", "President", "Knight"]
rank_colors = {
    "King": "#FFD700", "Duke": "#4169E1", "Prince": "#9370DB",
    "Marquis": "#2E8B57", "Earl": "#DC143C", "President": "#FF8C00",
    "Knight": "#808080"
}

rank_features = {r: {"fractal_dim": [], "junctions": [], "endpoints": [],
                      "holes": [], "ink_ratio": [], "h_symmetry": [],
                      "components": [], "skeleton_len": [], "n_lines": [],
                      "n_circles": [], "cluster": []}
                 for r in RANKS}

for d in demons:
    sid = d["id"]
    rank = d["rank"]
    if sid in feat_map and sid in skel_map:
        f = feat_map[sid]
        s = skel_map[sid]
        h = hough_map.get(sid, {})
        rank_features[rank]["fractal_dim"].append(f["fractal_dimension"])
        rank_features[rank]["junctions"].append(s["junctions"])
        rank_features[rank]["endpoints"].append(s["endpoints"])
        rank_features[rank]["holes"].append(s["holes"])
        rank_features[rank]["ink_ratio"].append(f["ink_ratio"])
        rank_features[rank]["h_symmetry"].append(f["horizontal_symmetry"])
        rank_features[rank]["components"].append(s["connected_components"])
        rank_features[rank]["skeleton_len"].append(s["skeleton_length_px"])
        rank_features[rank]["n_lines"].append(h.get("n_lines", 0))
        rank_features[rank]["n_circles"].append(h.get("n_circles", 0))
        rank_features[rank]["cluster"].append(sigil_to_cluster.get(sid, 0))


# ============================================================
# ANALYSIS 2: Legions vs Complexity
# ============================================================
legions_list = []
fd_list = []
junc_list = []
ink_list = []
holes_list = []
rank_list = []

for d in demons:
    sid = d["id"]
    if sid in feat_map and sid in skel_map:
        legions_list.append(d["legions"])
        fd_list.append(feat_map[sid]["fractal_dimension"])
        junc_list.append(skel_map[sid]["junctions"])
        ink_list.append(feat_map[sid]["ink_ratio"])
        holes_list.append(skel_map[sid]["holes"])
        rank_list.append(d["rank"])


# ============================================================
# ANALYSIS 3: Ability categories vs features
# ============================================================
# Categorize abilities
ability_categories = {
    "divination": ["divination", "past/present/future", "past/future", "hidden things", "future"],
    "love": ["cause love", "love of women", "friendship"],
    "destruction": ["kill", "drown", "destroy", "burn", "storms", "discord", "gangrene", "afflict"],
    "knowledge": ["liberal sciences", "arts and sciences", "philosophy", "astronomy", "geometry",
                   "rhetoric", "logic", "herbs", "precious stones", "teach", "sciences"],
    "transformation": ["invisibility", "shapeshift", "transform", "transmute"],
    "military": ["build towers", "warships", "battle", "weapons", "ammunition", "warrior", "fighter"],
    "wealth": ["treasure", "riches", "gold", "steal", "money", "dignities"],
    "familiar": ["familiar", "familiars"],
}

demon_categories = {d["id"]: set() for d in demons}
for d in demons:
    abilities_str = " ".join(d["abilities"]).lower()
    for cat, keywords in ability_categories.items():
        for kw in keywords:
            if kw.lower() in abilities_str:
                demon_categories[d["id"]].add(cat)
                break

# Compute average features per ability category
cat_features = {}
for cat in ability_categories:
    members = [sid for sid, cats in demon_categories.items() if cat in cats]
    if not members:
        continue
    fds = [feat_map[s]["fractal_dimension"] for s in members if s in feat_map]
    inks = [feat_map[s]["ink_ratio"] for s in members if s in feat_map]
    juncs = [skel_map[s]["junctions"] for s in members if s in skel_map]
    holes = [skel_map[s]["holes"] for s in members if s in skel_map]
    h_syms = [feat_map[s]["horizontal_symmetry"] for s in members if s in feat_map]
    cat_features[cat] = {
        "n": len(members),
        "avg_fd": np.mean(fds),
        "avg_ink": np.mean(inks),
        "avg_junctions": np.mean(juncs),
        "avg_holes": np.mean(holes),
        "avg_h_symmetry": np.mean(h_syms),
    }


# ============================================================
# ANALYSIS 4: Appearance type vs features
# ============================================================
# Categorize appearances
appearance_types = {
    "human": ["man", "woman", "soldier", "knight", "boy", "angel", "old man", "warrior", "archer"],
    "beast": ["lion", "wolf", "horse", "bull", "dog", "cat", "leopard", "bear", "toad"],
    "bird": ["raven", "crow", "owl", "stork", "phoenix", "peacock", "thrush", "griffin"],
    "serpent": ["serpent", "dragon", "viper"],
    "aquatic": ["mermaid", "sea monster", "crocodile"],
    "chimera": ["three heads", "griffin wings", "multiple", "monster"],
}

demon_appearance = {d["id"]: set() for d in demons}
for d in demons:
    app_str = " ".join(d["appearance"]).lower()
    for atype, keywords in appearance_types.items():
        for kw in keywords:
            if kw.lower() in app_str:
                demon_appearance[d["id"]].add(atype)
                break


# ============================================================
# VISUALIZATION
# ============================================================

fig = plt.figure(figsize=(22, 28))

# --- Plot 1: Rank vs Fractal Dimension (box plot) ---
ax1 = fig.add_subplot(4, 2, 1)
rank_data_fd = [rank_features[r]["fractal_dim"] for r in RANKS if rank_features[r]["fractal_dim"]]
rank_labels_fd = [f"{r}\n(n={len(rank_features[r]['fractal_dim'])})"
                  for r in RANKS if rank_features[r]["fractal_dim"]]
rank_colors_fd = [rank_colors[r] for r in RANKS if rank_features[r]["fractal_dim"]]

bp1 = ax1.boxplot(rank_data_fd, labels=rank_labels_fd, patch_artist=True, widths=0.6)
for patch, color in zip(bp1['boxes'], rank_colors_fd):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax1.set_ylabel("Fractal Dimension")
ax1.set_title("Sigil Complexity by Demon Rank", fontsize=12, fontweight='bold')
ax1.axhline(np.mean(fd_list), color='gray', linestyle='--', alpha=0.5, label='Overall mean')
ax1.legend(fontsize=8)

# --- Plot 2: Rank vs Ink Ratio ---
ax2 = fig.add_subplot(4, 2, 2)
rank_data_ink = [rank_features[r]["ink_ratio"] for r in RANKS if rank_features[r]["ink_ratio"]]
bp2 = ax2.boxplot(rank_data_ink, labels=rank_labels_fd, patch_artist=True, widths=0.6)
for patch, color in zip(bp2['boxes'], rank_colors_fd):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax2.set_ylabel("Ink Ratio")
ax2.set_title("Sigil Density by Demon Rank", fontsize=12, fontweight='bold')
ax2.axhline(np.mean(ink_list), color='gray', linestyle='--', alpha=0.5)

# --- Plot 3: Legions vs Fractal Dimension scatter ---
ax3 = fig.add_subplot(4, 2, 3)
for r in RANKS:
    mask = [rank_list[i] == r for i in range(len(rank_list))]
    leg_r = [legions_list[i] for i in range(len(mask)) if mask[i]]
    fd_r = [fd_list[i] for i in range(len(mask)) if mask[i]]
    ax3.scatter(leg_r, fd_r, c=rank_colors[r], label=r, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

r_val, p_val = stats.pearsonr(legions_list, fd_list)
ax3.set_xlabel("Number of Legions")
ax3.set_ylabel("Fractal Dimension")
ax3.set_title(f"Legions vs Sigil Complexity (r={r_val:.3f}, p={p_val:.3f})", fontsize=12, fontweight='bold')
ax3.legend(fontsize=7, ncol=2)

# --- Plot 4: Legions vs Junctions ---
ax4 = fig.add_subplot(4, 2, 4)
for r in RANKS:
    mask = [rank_list[i] == r for i in range(len(rank_list))]
    leg_r = [legions_list[i] for i in range(len(mask)) if mask[i]]
    junc_r = [junc_list[i] for i in range(len(mask)) if mask[i]]
    ax4.scatter(leg_r, junc_r, c=rank_colors[r], label=r, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

r_val2, p_val2 = stats.pearsonr(legions_list, junc_list)
ax4.set_xlabel("Number of Legions")
ax4.set_ylabel("Junctions (branch points)")
ax4.set_title(f"Legions vs Branch Complexity (r={r_val2:.3f}, p={p_val2:.3f})", fontsize=12, fontweight='bold')
ax4.legend(fontsize=7, ncol=2)

# --- Plot 5: Ability Category Feature Profiles ---
ax5 = fig.add_subplot(4, 2, 5)
cats = sorted(cat_features.keys(), key=lambda c: -cat_features[c]["n"])
x = np.arange(len(cats))
width = 0.35

# Normalize features for comparison
all_fd = [cat_features[c]["avg_fd"] for c in cats]
all_junc = [cat_features[c]["avg_junctions"] for c in cats]
all_holes = [cat_features[c]["avg_holes"] for c in cats]

bars1 = ax5.bar(x - width/2, all_fd, width, label='Avg Fractal Dim', color='steelblue', alpha=0.8)
ax5_twin = ax5.twinx()
bars2 = ax5_twin.bar(x + width/2, all_junc, width, label='Avg Junctions', color='coral', alpha=0.8)

ax5.set_xticks(x)
ax5.set_xticklabels([f"{c}\n(n={cat_features[c]['n']})" for c in cats], rotation=30, ha='right', fontsize=8)
ax5.set_ylabel("Fractal Dimension", color='steelblue')
ax5_twin.set_ylabel("Junctions", color='coral')
ax5.set_title("Sigil Features by Demon Ability Type", fontsize=12, fontweight='bold')
ax5.legend(loc='upper left', fontsize=8)
ax5_twin.legend(loc='upper right', fontsize=8)

# --- Plot 6: Appearance Type vs Cluster ---
ax6 = fig.add_subplot(4, 2, 6)
app_cluster_data = {}
for atype in appearance_types:
    members = [sid for sid, apps in demon_appearance.items() if atype in apps]
    cluster_counts = Counter(sigil_to_cluster.get(s, 0) for s in members if s in sigil_to_cluster)
    app_cluster_data[atype] = cluster_counts

atypes = sorted(app_cluster_data.keys(), key=lambda a: -sum(app_cluster_data[a].values()))
cluster_ids = sorted(set(sigil_to_cluster.values()))
cluster_colors_map = {1: '#E67E22', 2: '#2ECC71', 3: '#3498DB', 4: '#9B59B6',
                      5: '#E74C3C', 6: '#1ABC9C', 7: '#F39C12', 8: '#95A5A6'}

bottoms = np.zeros(len(atypes))
for cl in cluster_ids:
    vals = [app_cluster_data[a].get(cl, 0) for a in atypes]
    ax6.bar(range(len(atypes)), vals, bottom=bottoms,
            color=cluster_colors_map.get(cl, '#333'), label=f'Cl.{cl}', alpha=0.8)
    bottoms += vals

ax6.set_xticks(range(len(atypes)))
ax6.set_xticklabels(atypes, rotation=30, ha='right', fontsize=9)
ax6.set_ylabel("Count")
ax6.set_title("Appearance Type → Sigil Cluster", fontsize=12, fontweight='bold')
ax6.legend(fontsize=7, ncol=4, title="Cluster")

# --- Plot 7: Rank distribution across clusters ---
ax7 = fig.add_subplot(4, 2, 7)
rank_cluster = {}
for d in demons:
    sid = d["id"]
    rank = d["rank"]
    cl = sigil_to_cluster.get(sid, 0)
    rank_cluster.setdefault(rank, Counter())[cl] += 1

ranks_ordered = [r for r in RANKS if r in rank_cluster]
bottoms = np.zeros(len(ranks_ordered))
for cl in cluster_ids:
    vals = [rank_cluster[r].get(cl, 0) for r in ranks_ordered]
    ax7.bar(range(len(ranks_ordered)), vals, bottom=bottoms,
            color=cluster_colors_map.get(cl, '#333'), label=f'Cl.{cl}', alpha=0.8)
    bottoms += vals

ax7.set_xticks(range(len(ranks_ordered)))
ax7.set_xticklabels([f"{r}\n(n={sum(rank_cluster[r].values())})" for r in ranks_ordered], fontsize=9)
ax7.set_ylabel("Count")
ax7.set_title("Demon Rank → Sigil Cluster Distribution", fontsize=12, fontweight='bold')
ax7.legend(fontsize=7, ncol=4, title="Cluster")

# --- Plot 8: Goetia ordering vs fractal dimension ---
ax8 = fig.add_subplot(4, 2, 8)
order_ids = sorted(feat_map.keys())
order_fds = [feat_map[s]["fractal_dimension"] for s in order_ids]
order_ranks = [demon_map[s]["rank"] for s in order_ids if s in demon_map]
order_colors = [rank_colors.get(demon_map[s]["rank"], '#333') for s in order_ids if s in demon_map]

ax8.bar(order_ids, order_fds, color=order_colors, alpha=0.7, width=0.8)
ax8.set_xlabel("Goetia Number (traditional ordering)")
ax8.set_ylabel("Fractal Dimension")
ax8.set_title("Sigil Complexity Across the Goetia Sequence", fontsize=12, fontweight='bold')
# Add legend for ranks
legend_elements = [Patch(facecolor=rank_colors[r], alpha=0.7, label=r) for r in RANKS]
ax8.legend(handles=legend_elements, fontsize=7, ncol=4, loc='upper right')

# Rolling average
window = 7
if len(order_fds) > window:
    rolling = np.convolve(order_fds, np.ones(window)/window, mode='valid')
    ax8.plot(range(window//2, len(rolling)+window//2), rolling, 'k-', linewidth=2,
             alpha=0.7, label=f'{window}-sigil moving avg')
    ax8.legend(fontsize=7, ncol=5, loc='upper right')

plt.tight_layout()
plt.savefig(str(OUTDIR / "textual_structural_correlation.png"), dpi=150, facecolor='white')
plt.close()
print("Saved textual_structural_correlation.png")


# ============================================================
# STATISTICAL TESTS
# ============================================================
print("\n" + "="*60)
print("STATISTICAL ANALYSIS: RANK vs SIGIL STRUCTURE")
print("="*60)

# Kruskal-Wallis test (non-parametric ANOVA) for rank vs fractal dimension
rank_groups_fd = [rank_features[r]["fractal_dim"] for r in RANKS if len(rank_features[r]["fractal_dim"]) >= 2]
rank_names_fd = [r for r in RANKS if len(rank_features[r]["fractal_dim"]) >= 2]
if len(rank_groups_fd) >= 2:
    h_stat, p_kw = stats.kruskal(*rank_groups_fd)
    print(f"\nKruskal-Wallis test (rank → fractal dimension):")
    print(f"  H={h_stat:.3f}, p={p_kw:.4f}")
    if p_kw < 0.05:
        print(f"  ** SIGNIFICANT at α=0.05 — rank is associated with sigil complexity **")
    else:
        print(f"  Not significant at α=0.05 — no evidence rank determines complexity")

# Same for junctions
rank_groups_j = [rank_features[r]["junctions"] for r in RANKS if len(rank_features[r]["junctions"]) >= 2]
if len(rank_groups_j) >= 2:
    h_stat, p_kw = stats.kruskal(*rank_groups_j)
    print(f"\nKruskal-Wallis test (rank → junctions):")
    print(f"  H={h_stat:.3f}, p={p_kw:.4f}")

# Same for holes
rank_groups_h = [rank_features[r]["holes"] for r in RANKS if len(rank_features[r]["holes"]) >= 2]
if len(rank_groups_h) >= 2:
    h_stat, p_kw = stats.kruskal(*rank_groups_h)
    print(f"\nKruskal-Wallis test (rank → holes):")
    print(f"  H={h_stat:.3f}, p={p_kw:.4f}")

# Rank mean summaries
print(f"\n{'Rank':<12} {'n':>3} {'Avg FD':>8} {'Avg Ink':>8} {'Avg Junc':>9} {'Avg Holes':>10} {'Avg Sym':>8}")
print("-" * 62)
for r in RANKS:
    rf = rank_features[r]
    n = len(rf["fractal_dim"])
    if n > 0:
        print(f"{r:<12} {n:>3} {np.mean(rf['fractal_dim']):>8.3f} "
              f"{np.mean(rf['ink_ratio']):>8.3f} {np.mean(rf['junctions']):>9.1f} "
              f"{np.mean(rf['holes']):>10.1f} {np.mean(rf['h_symmetry']):>8.3f}")

print(f"\n{'='*60}")
print("CORRELATION: LEGIONS vs SIGIL FEATURES")
print("="*60)

for name, values in [("fractal_dim", fd_list), ("junctions", junc_list),
                      ("ink_ratio", ink_list), ("holes", holes_list)]:
    r_val, p_val = stats.pearsonr(legions_list, values)
    r_s, p_s = stats.spearmanr(legions_list, values)
    sig = "**" if p_val < 0.05 else ""
    print(f"  Legions vs {name:<15}: Pearson r={r_val:>7.3f} (p={p_val:.4f}) {sig}  "
          f"Spearman ρ={r_s:>7.3f} (p={p_s:.4f})")

print(f"\n{'='*60}")
print("ABILITY CATEGORY PROFILES")
print("="*60)
print(f"\n{'Category':<16} {'n':>3} {'Avg FD':>8} {'Avg Ink':>8} {'Avg Junc':>9} {'Avg Holes':>10} {'Avg Sym':>8}")
print("-" * 65)
for cat in cats:
    cf = cat_features[cat]
    print(f"{cat:<16} {cf['n']:>3} {cf['avg_fd']:>8.3f} {cf['avg_ink']:>8.3f} "
          f"{cf['avg_junctions']:>9.1f} {cf['avg_holes']:>10.1f} {cf['avg_h_symmetry']:>8.3f}")

# Chi-squared test: is rank distribution across clusters non-random?
print(f"\n{'='*60}")
print("CHI-SQUARED: RANK → CLUSTER INDEPENDENCE")
print("="*60)

# Build contingency table
rank_cl_table = []
for r in ranks_ordered:
    row = [rank_cluster[r].get(cl, 0) for cl in cluster_ids]
    rank_cl_table.append(row)

rank_cl_array = np.array(rank_cl_table)
# Only test if enough data (merge small groups)
if rank_cl_array.sum() > 0:
    chi2, p_chi, dof, expected = stats.chi2_contingency(rank_cl_array)
    print(f"  χ²={chi2:.2f}, df={dof}, p={p_chi:.4f}")
    if p_chi < 0.05:
        print(f"  ** SIGNIFICANT — demon rank is NOT independent of sigil cluster **")
    else:
        print(f"  Not significant — rank appears independent of structural family")

# Goetia ordering trend
print(f"\n{'='*60}")
print("TREND: GOETIA ORDERING vs COMPLEXITY")
print("="*60)
r_order, p_order = stats.spearmanr(order_ids, order_fds)
print(f"  Spearman correlation (Goetia # → FD): ρ={r_order:.3f}, p={p_order:.4f}")
if p_order < 0.05:
    direction = "increases" if r_order > 0 else "decreases"
    print(f"  ** SIGNIFICANT — sigil complexity {direction} through the sequence **")
else:
    print(f"  Not significant — no trend in complexity across the ordering")

# Save all results
results = {
    "rank_feature_means": {r: {k: round(float(np.mean(v)), 4) if v else None
                                for k, v in rank_features[r].items() if k != "cluster"}
                           for r in RANKS},
    "legions_correlations": {
        "vs_fractal_dim": {"pearson_r": round(float(stats.pearsonr(legions_list, fd_list)[0]), 4),
                           "pearson_p": round(float(stats.pearsonr(legions_list, fd_list)[1]), 4)},
        "vs_junctions": {"pearson_r": round(float(stats.pearsonr(legions_list, junc_list)[0]), 4),
                         "pearson_p": round(float(stats.pearsonr(legions_list, junc_list)[1]), 4)},
    },
    "ability_category_profiles": cat_features,
    "ordering_trend": {"spearman_rho": round(float(r_order), 4),
                       "p_value": round(float(p_order), 4)},
}

with open(OUTDIR / "textual_correlation_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nSaved results to textual_correlation_results.json")
