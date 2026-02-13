"""
Script 2: Skeleton/Thinning + Connected Component Analysis
Reduces each sigil to a 1-pixel skeleton and analyzes topology.
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize
from pathlib import Path
import json

SIGIL_DIR = Path(r"C:\Users\PC\Downloads\goetia_analysis\extracted_sigils")
OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis\skeletons")
OUTDIR.mkdir(parents=True, exist_ok=True)

metadata_file = SIGIL_DIR.parent / "sigil_metadata.json"
if not metadata_file.exists():
    print("Run 01_segment_sigils.py first!")
    exit(1)

with open(metadata_file) as f:
    sigils = json.load(f)

results = []

for info in sigils:
    fpath = SIGIL_DIR / info["file"]
    if not fpath.exists():
        continue

    img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)

    # Binarize
    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)

    # Clean up small noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Skeletonize
    skel = skeletonize(binary > 0).astype(np.uint8) * 255

    # Save skeleton
    cv2.imwrite(str(OUTDIR / info["file"]), skel)

    # Connected components on original binary
    num_components, labels = cv2.connectedComponents(binary)

    # Count skeleton pixels (total stroke length proxy)
    skel_pixels = np.sum(skel > 0)

    # Find junctions (pixels with >2 neighbors in skeleton)
    # and endpoints (pixels with exactly 1 neighbor)
    padded = np.pad(skel // 255, 1, mode='constant')
    neighbor_count = np.zeros_like(skel, dtype=int)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            neighbor_count += padded[1+dy:padded.shape[0]-1+dy, 1+dx:padded.shape[1]-1+dx]

    skel_mask = skel > 0
    junctions = np.sum((neighbor_count > 2) & skel_mask)
    endpoints = np.sum((neighbor_count == 1) & skel_mask)

    # Euler number: C - H (connected components minus holes)
    # For binary image
    contours_outer, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_ext, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    holes = len(contours_outer) - len(contours_ext)
    euler = (num_components - 1) - holes  # subtract 1 for background

    result = {
        "id": info["id"],
        "connected_components": int(num_components - 1),
        "skeleton_length_px": int(skel_pixels),
        "junctions": int(junctions),
        "endpoints": int(endpoints),
        "holes": int(holes),
        "euler_number": int(euler),
        "junction_endpoint_ratio": round(junctions / max(endpoints, 1), 3),
    }
    results.append(result)

    if info["id"] <= 5:  # Print first few
        print(f"Sigil {info['id']}: components={result['connected_components']}, "
              f"skel_len={result['skeleton_length_px']}, junctions={result['junctions']}, "
              f"endpoints={result['endpoints']}, holes={result['holes']}, euler={result['euler_number']}")

with open(SIGIL_DIR.parent / "skeleton_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

# Summary statistics
if results:
    print(f"\n=== Summary across {len(results)} sigils ===")
    for key in ["connected_components", "junctions", "endpoints", "holes", "euler_number"]:
        vals = [r[key] for r in results]
        print(f"{key}: min={min(vals)}, max={max(vals)}, mean={np.mean(vals):.1f}, std={np.std(vals):.1f}")
