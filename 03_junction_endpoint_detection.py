"""
Script 3: Junction & Endpoint Detection with Visualization
Finds where lines branch (junctions) and where they terminate (endpoints).
Classifies terminal decorations.
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize
from pathlib import Path
import json

SIGIL_DIR = Path(r"C:\Users\PC\Downloads\goetia_analysis\extracted_sigils")
SKEL_DIR = Path(r"C:\Users\PC\Downloads\goetia_analysis\skeletons")
OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis\junction_maps")
OUTDIR.mkdir(parents=True, exist_ok=True)

metadata_file = SIGIL_DIR.parent / "sigil_metadata.json"
with open(metadata_file) as f:
    sigils = json.load(f)

def get_neighbor_count(skel):
    """Count 8-connected neighbors for each skeleton pixel."""
    padded = np.pad(skel.astype(np.float32), 1, mode='constant')
    nc = np.zeros_like(skel, dtype=np.float32)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            nc += padded[1+dy:padded.shape[0]-1+dy, 1+dx:padded.shape[1]-1+dx]
    return nc

def classify_endpoint_region(binary, ey, ex, radius=15):
    """Classify what kind of terminal decoration is near an endpoint."""
    h, w = binary.shape
    y1, y2 = max(0, ey-radius), min(h, ey+radius)
    x1, x2 = max(0, ex-radius), min(w, ex+radius)
    region = binary[y1:y2, x1:x2]

    if region.size == 0:
        return "unknown"

    # Check for circle: find contours in region
    contours, _ = cv2.findContours(region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.7 and area > 20:
                return "circle"

    # Check for cross/plus shape (high pixel density in both axes)
    ink_ratio = np.sum(region > 0) / region.size
    if ink_ratio > 0.3:
        return "filled/cross"

    # Check for bar/line terminal
    rows_with_ink = np.sum(np.any(region > 0, axis=1))
    cols_with_ink = np.sum(np.any(region > 0, axis=0))
    if rows_with_ink < 5 and cols_with_ink > 8:
        return "horizontal_bar"
    if cols_with_ink < 5 and rows_with_ink > 8:
        return "vertical_bar"

    return "simple"

all_terminal_types = {}
junction_stats = []

for info in sigils[:72]:  # Process all
    fpath = SIGIL_DIR / info["file"]
    if not fpath.exists():
        continue

    img = cv2.imread(str(fpath))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    skel = skeletonize(binary > 0).astype(np.uint8)
    nc = get_neighbor_count(skel)

    skel_mask = skel > 0
    junction_pts = np.argwhere((nc > 2) & skel_mask)  # [y, x]
    endpoint_pts = np.argwhere((nc == 1) & skel_mask)

    # Cluster nearby junctions (they often form small groups)
    if len(junction_pts) > 0:
        from scipy.ndimage import label as scipy_label
        junc_img = ((nc > 2) & skel_mask).astype(np.uint8)
        junc_dilated = cv2.dilate(junc_img, np.ones((5,5), np.uint8))
        _, n_junction_clusters = scipy_label(junc_dilated)
    else:
        n_junction_clusters = 0

    # Classify endpoints
    terminal_types = []
    for ey, ex in endpoint_pts:
        tt = classify_endpoint_region(binary, ey, ex)
        terminal_types.append(tt)

    # Visualize
    vis = img.copy()
    for ey, ex in junction_pts:
        cv2.circle(vis, (ex, ey), 3, (0, 0, 255), -1)  # Red = junction
    for ey, ex in endpoint_pts:
        cv2.circle(vis, (ex, ey), 3, (0, 255, 0), -1)  # Green = endpoint

    cv2.imwrite(str(OUTDIR / info["file"]), vis)

    # Count terminal types
    type_counts = {}
    for tt in terminal_types:
        type_counts[tt] = type_counts.get(tt, 0) + 1

    junction_stats.append({
        "id": info["id"],
        "raw_junctions": len(junction_pts),
        "junction_clusters": int(n_junction_clusters),
        "endpoints": len(endpoint_pts),
        "terminal_types": type_counts
    })

    for tt in type_counts:
        all_terminal_types[tt] = all_terminal_types.get(tt, 0) + type_counts[tt]

with open(SIGIL_DIR.parent / "junction_analysis.json", "w") as f:
    json.dump(junction_stats, f, indent=2)

print(f"Processed {len(junction_stats)} sigils")
print(f"\n=== Terminal decoration types across all sigils ===")
for tt, count in sorted(all_terminal_types.items(), key=lambda x: -x[1]):
    print(f"  {tt}: {count}")

print(f"\n=== Junction cluster stats ===")
clusters = [s["junction_clusters"] for s in junction_stats]
print(f"  min={min(clusters)}, max={max(clusters)}, mean={np.mean(clusters):.1f}")
