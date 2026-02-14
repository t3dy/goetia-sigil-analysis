"""
Script 18: Re-segment Sigils (v3.0)
Include each demon's correct English name label ABOVE the seal,
while excluding the wrong name label from the row below.

v2 stripped both top and bottom text, leaving just the seal drawing.
v3 keeps the correct name visible above the seal for context.

Strategy:
1. Same 10x8 grid as v2
2. For each cell, find the correct name text at the top (~first 35px)
3. Find where the seal drawing ends
4. Find where the NEXT row's text starts (from the bottom)
5. Extract from top-of-text to end-of-seal, excluding next row's text
"""
import cv2
import numpy as np
from pathlib import Path
import json
import sys
import shutil

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")
INPUT = Path(r"C:\Users\PC\Downloads\1280px-72_Goeta_sigils.png")
NEW_SIGIL_DIR = OUTDIR / "extracted_sigils_v3"

# Clean start
if NEW_SIGIL_DIR.exists():
    shutil.rmtree(NEW_SIGIL_DIR)
NEW_SIGIL_DIR.mkdir()

DEMON_NAMES = [
    "Bael","Agares","Vassago","Samigina","Marbas","Valefor","Amon",
    "Barbatos","Paimon","Buer","Gusion","Sitri","Beleth","Leraje",
    "Eligos","Zepar","Botis","Bathin","Sallos","Purson","Marax",
    "Ipos","Aim","Naberius","Glasya-Labolas","Bune","Ronove",
    "Berith","Astaroth","Forneus","Foras","Asmoday","Gaap",
    "Furfur","Marchosias","Stolas","Phenex","Halphas","Malphas",
    "Raum","Focalor","Vepar","Sabnock","Shax","Vine","Bifrons",
    "Uvall","Haagenti","Crocell","Furcas","Balam","Alloces",
    "Camio","Murmur","Orobas","Gremory","Ose","Amy","Oriax",
    "Vapula","Zagan","Volac","Andras","Haures","Andrealphus",
    "Cimejes","Amdusias","Belial","Decarabia","Seere","Dantalion",
    "Andromalius"
]

img = cv2.imread(str(INPUT), cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(str(INPUT))
h, w = img.shape
print(f"Image size: {w}x{h}")

# Grid layout (same as v2)
col_edges = [0, 160, 280, 400, 525, 645, 760, 875, 995, 1115, w]
row_edges = [0, 145, 260, 370, 485, 600, 720, 830, h]

n_cols = len(col_edges) - 1
n_rows = len(row_edges) - 1

# Grid mapping (same as v2)
grid_map = {}
for col in range(9):
    grid_map[(0, col)] = col + 1
grid_map[(0, 9)] = None

for col in range(9):
    grid_map[(1, col)] = 10 + col
grid_map[(1, 9)] = None

for row_idx in range(2, 7):
    for col in range(10):
        grid_map[(row_idx, col)] = 19 + (row_idx - 2) * 10 + col

for col in range(4):
    grid_map[(7, col)] = 69 + col
for col in range(4, 10):
    grid_map[(7, col)] = None


def extract_sigil_with_label(gray, x1, y1, x2, y2, padding=3):
    """Extract the sigil INCLUDING its correct name label above it,
    but EXCLUDING the next row's name label below.

    Layout within each cell:
    - Top: Correct name label (e.g. "1. Bael") - typically y=15-35 within cell
    - Middle: The seal drawing
    - Bottom: Next row's name label bleeding in - typically last 15px

    We want to include from the start of the correct label to the end of the seal.
    """
    cell = gray[y1:y2, x1:x2]
    cell_h, cell_w = cell.shape

    # Binarize
    _, binary = cv2.threshold(cell, 180, 255, cv2.THRESH_BINARY_INV)

    row_ink = np.sum(binary > 0, axis=1)

    # Find the first ink (this is the start of the name label)
    first_ink_y = None
    for y in range(cell_h):
        if row_ink[y] > 0:
            first_ink_y = y
            break

    if first_ink_y is None:
        return None, None

    # Now scan from the bottom to find where the NEXT ROW's text starts
    # We look for a gap between the seal and the bottom text
    # The bottom text is typically in the last ~15px of the cell

    # Find the last ink from the bottom
    last_ink_y = None
    for y in range(cell_h - 1, -1, -1):
        if row_ink[y] > 0:
            last_ink_y = y
            break

    if last_ink_y is None:
        return None, None

    # Now detect if there's a gap near the bottom separating seal from next-row text
    # Strategy: scan upward from the bottom, find the first gap of 3+ empty rows
    # Everything below that gap is next-row text and should be excluded
    cut_bottom = last_ink_y + 1  # default: include everything

    # Only search in the bottom portion of the cell
    bottom_search_start = max(last_ink_y - 30, first_ink_y + 20)

    # Find gaps (consecutive rows with 0 ink) scanning from bottom
    scan_y = last_ink_y
    while scan_y > bottom_search_start:
        if row_ink[scan_y] == 0:
            # Found an empty row - count how many consecutive empty rows
            gap_end = scan_y
            gap_start = scan_y
            while gap_start > bottom_search_start and row_ink[gap_start - 1] == 0:
                gap_start -= 1
            gap_size = gap_end - gap_start + 1

            # If gap is 3+ rows, this separates seal from next-row text
            if gap_size >= 3:
                # Check that the ink below the gap is small (text-sized)
                ink_below_gap = last_ink_y - gap_end
                if ink_below_gap > 0 and ink_below_gap < 20:
                    cut_bottom = gap_start  # cut at the start of the gap
                    break

            scan_y = gap_start - 1
        else:
            scan_y -= 1

    # Also check for a gap near the top to separate text from seal
    # But we WANT to keep the text, so we DON'T cut it
    # However, we should detect it to know where the seal starts

    # Find ink extents in x
    safe_binary = binary[first_ink_y:cut_bottom, :]
    coords = np.where(safe_binary > 0)
    if len(coords[0]) == 0:
        return None, None

    min_x, max_x = coords[1].min(), coords[1].max()

    # Add padding (minimal on bottom to avoid next-row text)
    extract_y1 = max(0, first_ink_y - padding)
    extract_y2 = min(cell_h, cut_bottom + 1)  # tight bottom crop, no padding
    extract_x1 = max(0, min_x - padding)
    extract_x2 = min(cell_w, max_x + padding)

    cropped = cell[extract_y1:extract_y2, extract_x1:extract_x2]

    global_bbox = (int(x1 + extract_x1), int(y1 + extract_y1),
                   int(extract_x2 - extract_x1), int(extract_y2 - extract_y1))
    return cropped, global_bbox


results = []
extracted_count = 0

for row in range(n_rows):
    for col in range(n_cols):
        demon_num = grid_map.get((row, col))

        x1 = col_edges[col]
        x2 = col_edges[col + 1]
        y1 = row_edges[row]
        y2 = row_edges[row + 1]

        if demon_num is None:
            continue

        if demon_num > 72:
            continue

        demon_name = DEMON_NAMES[demon_num - 1]

        sigil_img, bbox = extract_sigil_with_label(img, x1, y1, x2, y2)

        if sigil_img is None:
            print(f"  WARNING: No ink found for #{demon_num} {demon_name} at row={row} col={col}")
            continue

        filename = f"sigil_{demon_num:03d}.png"
        cv2.imwrite(str(NEW_SIGIL_DIR / filename), sigil_img)

        results.append({
            "id": demon_num,
            "name": demon_name,
            "bbox": list(bbox),
            "file": filename,
            "aspect_ratio": round(bbox[2] / bbox[3], 3) if bbox[3] > 0 else 0,
            "grid_row": row,
            "grid_col": col
        })

        extracted_count += 1
        if demon_num <= 5 or demon_num in [16, 17, 72]:
            print(f"  #{demon_num:3d} {demon_name:20s} row={row} col={col} size={sigil_img.shape[1]}x{sigil_img.shape[0]}")

print(f"\nExtracted {extracted_count} sigils with name labels")

results.sort(key=lambda x: x["id"])

with open(OUTDIR / "sigil_metadata_v3.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved sigil_metadata_v3.json with {len(results)} entries")

# Verification
print("\n=== Verification Summary ===")
found_ids = set(r["id"] for r in results)
missing = [i for i in range(1, 73) if i not in found_ids]
if missing:
    print(f"MISSING demons: {missing}")
else:
    print("All 72 demons accounted for!")

print("\nAspect ratio check (>2.5 = suspicious):")
suspicious = [r for r in results if r["aspect_ratio"] > 2.5]
if suspicious:
    for r in suspicious:
        print(f"  #{r['id']} {r['name']}: aspect={r['aspect_ratio']:.2f}")
else:
    print("  No suspicious extractions")

# Create verification grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(8, 9, figsize=(27, 24))
fig.suptitle("V3.0 Seal Verification Grid (72 Demons - With Name Labels)", fontsize=18, fontweight='bold')

for r in results:
    sid = r["id"]
    if sid > 72:
        continue
    row_idx = (sid - 1) // 9
    col_idx = (sid - 1) % 9

    if row_idx >= 8:
        continue

    ax = axes[row_idx][col_idx]
    img_path = NEW_SIGIL_DIR / r["file"]
    if img_path.exists():
        sigil = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        ax.imshow(sigil, cmap='gray')
    ax.set_title(f"#{sid} {r['name']}", fontsize=7)
    ax.axis('off')

for row_idx in range(8):
    for col_idx in range(9):
        sid = row_idx * 9 + col_idx + 1
        if sid > 72:
            axes[row_idx][col_idx].axis('off')

plt.tight_layout()
plt.savefig(str(OUTDIR / "v3_verification_grid.png"), dpi=150)
plt.close()

print("\nSaved v3_verification_grid.png for visual inspection")
