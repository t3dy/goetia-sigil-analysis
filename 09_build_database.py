"""
Script 9: Build Unified Sigil Database
Merges all analysis results + textual metadata into a single comprehensive database.
Also encodes sigil images as base64 for embedding in the web dashboard.
"""
import json
import base64
import numpy as np
from pathlib import Path

OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")
SIGIL_DIR = OUTDIR / "extracted_sigils"
SKEL_DIR = OUTDIR / "skeletons"
JUNC_DIR = OUTDIR / "junction_maps"

# Load all data sources
with open(OUTDIR / "demon_metadata.json") as f:
    demons = json.load(f)
demon_map = {d["id"]: d for d in demons}

with open(OUTDIR / "sigil_metadata.json") as f:
    sigil_meta = json.load(f)
sigil_meta_map = {s["id"]: s for s in sigil_meta}

with open(OUTDIR / "features.json") as f:
    features = json.load(f)
feat_map = {f["id"]: f for f in features}

with open(OUTDIR / "skeleton_analysis.json") as f:
    skel_data = json.load(f)
skel_map = {s["id"]: s for s in skel_data}

with open(OUTDIR / "hough_analysis.json") as f:
    hough_data = json.load(f)
hough_map = {h["id"]: h for h in hough_data}

with open(OUTDIR / "junction_analysis.json") as f:
    junc_data = json.load(f)
junc_map = {j["id"]: j for j in junc_data}

with open(OUTDIR / "cluster_assignments.json") as f:
    clusters = json.load(f)

# Invert cluster map
sigil_to_cluster = {}
for cl, members in clusters.items():
    for m in members:
        sigil_to_cluster[m] = int(cl)

CLUSTER_NAMES = {
    1: "Sparse Minimalists",
    2: "Spread Networks",
    3: "Circuit Loops",
    4: "Diagonal Stars",
    5: "Mixed-Angle Composites",
    6: "Wide Horizontals",
    7: "Unified Verticals",
    8: "Outlier (Zepar)",
}

# Ability categories
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

def get_ability_cats(abilities):
    cats = set()
    abilities_str = " ".join(abilities).lower()
    for cat, keywords in ability_categories.items():
        for kw in keywords:
            if kw.lower() in abilities_str:
                cats.add(cat)
                break
    return sorted(cats)

def encode_image(path):
    if path.exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('ascii')
    return None

# Build unified records
database = []
for sid in range(1, 79):  # We extracted up to 78
    record = {"id": sid}

    # Textual metadata
    if sid in demon_map:
        d = demon_map[sid]
        record["name"] = d["name"]
        record["rank"] = d["rank"]
        record["legions"] = d["legions"]
        record["appearance"] = d["appearance"]
        record["abilities"] = d["abilities"]
        record["ability_categories"] = get_ability_cats(d["abilities"])
        record["element"] = d.get("element")
    else:
        record["name"] = f"Sigil {sid}"
        record["rank"] = None
        record["legions"] = None
        record["appearance"] = []
        record["abilities"] = []
        record["ability_categories"] = []
        record["element"] = None

    # Cluster assignment
    record["cluster_id"] = sigil_to_cluster.get(sid)
    record["cluster_name"] = CLUSTER_NAMES.get(sigil_to_cluster.get(sid, 0), "Unknown")

    # Geometric features
    if sid in feat_map:
        f = feat_map[sid]
        record["features"] = {
            "ink_ratio": f["ink_ratio"],
            "aspect_ratio": f["aspect_ratio"],
            "compactness": f["compactness"],
            "fractal_dimension": f["fractal_dimension"],
            "horizontal_symmetry": f["horizontal_symmetry"],
            "vertical_symmetry": f["vertical_symmetry"],
            "radial_profile": f["radial_profile"],
            "quadrant_density": f["quadrant_density"],
        }
    else:
        record["features"] = None

    # Skeleton topology
    if sid in skel_map:
        s = skel_map[sid]
        record["topology"] = {
            "connected_components": s["connected_components"],
            "skeleton_length_px": s["skeleton_length_px"],
            "junctions": s["junctions"],
            "endpoints": s["endpoints"],
            "holes": s["holes"],
            "euler_number": s["euler_number"],
            "junction_endpoint_ratio": s["junction_endpoint_ratio"],
        }
    else:
        record["topology"] = None

    # Hough geometry
    if sid in hough_map:
        h = hough_map[sid]
        record["geometry"] = {
            "n_lines": h["n_lines"],
            "n_circles": h["n_circles"],
            "dominant_angle": h["dominant_angle"],
            "angle_histogram": h["angle_histogram"],
            "line_circle_ratio": h["line_circle_ratio"],
        }
    else:
        record["geometry"] = None

    # Junction analysis
    if sid in junc_map:
        j = junc_map[sid]
        record["junction_detail"] = {
            "raw_junctions": j["raw_junctions"],
            "junction_clusters": j["junction_clusters"],
            "endpoints": j["endpoints"],
            "terminal_types": j["terminal_types"],
        }
    else:
        record["junction_detail"] = None

    # Encode images (only for valid sigils with demon data)
    if sid <= 72 and sid in demon_map:
        sigil_path = SIGIL_DIR / f"sigil_{sid:03d}.png"
        skel_path = SKEL_DIR / f"sigil_{sid:03d}.png"
        junc_path = JUNC_DIR / f"sigil_{sid:03d}.png"

        record["images"] = {
            "sigil": encode_image(sigil_path),
            "skeleton": encode_image(skel_path),
            "junction_map": encode_image(junc_path),
        }
    else:
        record["images"] = None

    # Only include demons 1-72 in final database
    if sid <= 72 and sid in demon_map:
        database.append(record)

# Save full database
with open(OUTDIR / "sigil_database.json", "w") as f:
    json.dump(database, f, indent=2)

# Save a lightweight version without images (for quick loading)
db_light = []
for rec in database:
    light = {k: v for k, v in rec.items() if k != "images"}
    db_light.append(light)

with open(OUTDIR / "sigil_database_light.json", "w") as f:
    json.dump(db_light, f, indent=2)

print(f"Built database with {len(database)} demons")
print(f"Full database: {OUTDIR / 'sigil_database.json'} ({(OUTDIR / 'sigil_database.json').stat().st_size / 1024:.0f} KB)")
print(f"Light database: {OUTDIR / 'sigil_database_light.json'} ({(OUTDIR / 'sigil_database_light.json').stat().st_size / 1024:.0f} KB)")

# Print summary
ranks = [r["rank"] for r in database if r["rank"]]
from collections import Counter
print(f"\nRank distribution: {dict(Counter(ranks))}")
clusters_dist = Counter(r["cluster_name"] for r in database)
print(f"Cluster distribution: {dict(clusters_dist)}")
all_cats = []
for r in database:
    all_cats.extend(r["ability_categories"])
print(f"Ability categories: {dict(Counter(all_cats))}")
