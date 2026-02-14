"""
Script 17: Re-segment Sigils (v2.0)
Completely redo the segmentation with correct demon-to-seal mapping.

The original image (1280x949) contains 72 seals + some variant/extra images
arranged in ~8 rows of ~10. The name label (e.g., "1. Bael") appears directly
above each seal.

Strategy:
1. Divide the image into a fixed grid (10 columns x 8 rows) based on visual layout
2. Within each cell, extract the sigil
3. Map cells to demon names using the known ordering
4. Handle edge cases: 9b. Paimon variant, merged regions, row 8 extras

From visual inspection of the 1280x949 image:
- 10 columns, each ~128px wide
- Row 1 (y~30-150): Demons 1-9 + Paimon variant
- Row 2 (y~145-265): Demons 10-18 + extra?
- Row 3 (y~255-370): Demons 19-28
- Row 4 (y~355-485): Demons 29-38
- Row 5 (y~480-600): Demons 39-48
- Row 6 (y~595-720): Demons 49-58
- Row 7 (y~710-835): Demons 59-68
- Row 8 (y~825-940): Demons 69-72 + extras
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
NEW_SIGIL_DIR = OUTDIR / "extracted_sigils_v2"

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

# ============================================================
# STEP 1: Define the grid layout
# ============================================================

# The image has roughly 10 columns
# Column boundaries (approximate x positions for each column center):
# Looking at the extracted bbox x-positions:
# Col 1: x~50-160   center ~105
# Col 2: x~165-280  center ~222
# Col 3: x~280-400  center ~340
# Col 4: x~400-525  center ~462
# Col 5: x~525-645  center ~585
# Col 6: x~645-760  center ~702
# Col 7: x~760-875  center ~817
# Col 8: x~875-995  center ~935
# Col 9: x~995-1115 center ~1055
# Col 10: x~1115-1230 center ~1172

# Column boundaries
col_edges = [0, 160, 280, 400, 525, 645, 760, 875, 995, 1115, w]
# Row boundaries (these separate the text-label regions from seal regions)
# Each "row" has: text at top, seal below
# Row boundaries (top of text region for each row)
row_edges = [0, 145, 260, 370, 485, 600, 720, 830, h]

n_cols = len(col_edges) - 1
n_rows = len(row_edges) - 1

print(f"Grid: {n_rows} rows x {n_cols} columns")

# ============================================================
# STEP 2: The mapping from grid position to demon number
# ============================================================

# Row 1: 1.Bael through 9.Paimon in cols 0-8, col 9 = "9b. Paimon" (variant, skip)
# Row 2: 10.Buer through 18.Bathin in cols 0-8, col 9 might be extra
# Row 3: 19.Sallos through 28.Berith in cols 0-9
# Row 4: 29.Astaroth through 38.Halphas in cols 0-9
# Row 5: 39.Malphas through 48.Haagenti in cols 0-9
# Row 6: 49.Crocell through 58.Amy in cols 0-9
# Row 7: 59.Oriax through 68.Belial in cols 0-9
# Row 8: 69.Decarabia through 72.Andromalius in cols 0-3, rest are extras

# Build the grid mapping: (row, col) -> demon_number (1-based) or None
grid_map = {}

# Row 1: demons 1-9 in columns 0-8
for col in range(9):
    grid_map[(0, col)] = col + 1  # 1-9
grid_map[(0, 9)] = None  # 9b variant

# Row 2: demons 10-18 in columns 0-8
for col in range(9):
    grid_map[(1, col)] = 10 + col  # 10-18
grid_map[(1, 9)] = None  # might be extra

# Row 3: demons 19-28 in columns 0-9
for col in range(10):
    grid_map[(2, col)] = 19 + col  # 19-28

# Row 4: demons 29-38 in columns 0-9
for col in range(10):
    grid_map[(3, col)] = 29 + col  # 29-38

# Row 5: demons 39-48 in columns 0-9
for col in range(10):
    grid_map[(4, col)] = 39 + col  # 39-48

# Row 6: demons 49-58 in columns 0-9
for col in range(10):
    grid_map[(5, col)] = 49 + col  # 49-58

# Row 7: demons 59-68 in columns 0-9
for col in range(10):
    grid_map[(6, col)] = 59 + col  # 59-68

# Row 8: demons 69-72 in columns 0-3
for col in range(4):
    grid_map[(7, col)] = 69 + col  # 69-72
for col in range(4, 10):
    grid_map[(7, col)] = None  # extras


# ============================================================
# STEP 3: Extract each cell
# ============================================================

def extract_sigil_from_cell(gray, x1, y1, x2, y2, padding=5):
    """Extract the sigil from a grid cell, trimming whitespace and text labels.

    Text labels appear in the top portion and bottom portion of each cell.
    We need to find and exclude them.
    Strategy: Skip the top 15px (where "N. Name" text lives from above)
              and bottom 15px (where next row text might bleed in).
              Then find the main ink blob.
    """
    cell = gray[y1:y2, x1:x2]
    cell_h, cell_w = cell.shape

    # Binarize the full cell
    _, binary = cv2.threshold(cell, 180, 255, cv2.THRESH_BINARY_INV)

    # Mask out the text regions:
    # The text label for THIS seal is at the very top of the cell
    # The text label for the NEXT ROW's seal is at the very bottom
    # We need to find where the actual seal lives.

    # Approach: scan horizontal ink density row by row.
    # Text lines are narrow (few pixels tall) with scattered ink.
    # The seal body is a contiguous region of moderate ink density.

    row_ink = np.sum(binary > 0, axis=1)  # ink per row

    # Find the main seal region: largest contiguous block of rows with ink
    # Skip top 10 pixels (text) and bottom 10 pixels (text)
    top_skip = 15
    bottom_skip = 15

    # Find bounding box of ink in the safe middle zone
    safe_binary = binary.copy()
    safe_binary[:top_skip, :] = 0  # mask top text
    safe_binary[cell_h - bottom_skip:, :] = 0  # mask bottom text

    coords = np.where(safe_binary > 0)
    if len(coords[0]) == 0:
        # Try with less aggressive masking
        safe_binary = binary.copy()
        safe_binary[:8, :] = 0
        safe_binary[cell_h - 8:, :] = 0
        coords = np.where(safe_binary > 0)
        if len(coords[0]) == 0:
            return None, None

    min_y, max_y = coords[0].min(), coords[0].max()
    min_x, max_x = coords[1].min(), coords[1].max()

    # Now check if there's leftover text at the bottom of the extracted region
    # Text typically spans < 12px in height and is at the very bottom
    # Check if the last 12 rows of the extracted area are separated from main body
    if max_y - min_y > 30:
        # Look for a gap (< 3 ink pixels in a row) near the bottom
        region_ink = np.sum(binary[min_y:max_y+1, min_x:max_x+1] > 0, axis=1)
        for scan_y in range(len(region_ink) - 1, max(len(region_ink) - 20, 0), -1):
            if region_ink[scan_y] == 0 and scan_y < len(region_ink) - 3:
                # Found a gap row near the bottom - trim to here
                # But only if this cuts off < 15px (text-sized)
                remaining = len(region_ink) - scan_y - 1
                if 3 < remaining < 18:
                    max_y = min_y + scan_y
                break

    # Similarly check for text at the top
    if max_y - min_y > 30:
        region_ink = np.sum(binary[min_y:max_y+1, min_x:max_x+1] > 0, axis=1)
        for scan_y in range(0, min(20, len(region_ink))):
            if region_ink[scan_y] == 0 and scan_y > 3:
                remaining = scan_y
                if 3 < remaining < 18:
                    min_y = min_y + scan_y + 1
                break

    # Add padding
    min_y = max(0, min_y - padding)
    max_y = min(cell_h, max_y + padding)
    min_x = max(0, min_x - padding)
    max_x = min(cell_w, max_x + padding)

    cropped = cell[min_y:max_y, min_x:max_x]

    # Return the cropped sigil and its global bbox
    global_bbox = (int(x1 + min_x), int(y1 + min_y), int(max_x - min_x), int(max_y - min_y))
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

        # Extract sigil
        sigil_img, bbox = extract_sigil_from_cell(img, x1, y1, x2, y2)

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

print(f"\nExtracted {extracted_count} sigils")

# Sort by id
results.sort(key=lambda x: x["id"])

# Save new metadata
with open(OUTDIR / "sigil_metadata_v2.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved sigil_metadata_v2.json with {len(results)} entries")

# ============================================================
# STEP 4: Verify by visual inspection of key sigils
# ============================================================

print("\n=== Verification Summary ===")
found_ids = set(r["id"] for r in results)
missing = [i for i in range(1, 73) if i not in found_ids]
if missing:
    print(f"MISSING demons: {missing}")
else:
    print("All 72 demons accounted for!")

# Check for any suspiciously wide extractions (merged seals)
print("\nAspect ratio check (>2.0 = suspicious):")
suspicious = [r for r in results if r["aspect_ratio"] > 2.0]
if suspicious:
    for r in suspicious:
        print(f"  #{r['id']} {r['name']}: aspect={r['aspect_ratio']:.2f}")
else:
    print("  No suspicious wide extractions - grid approach solved the merging!")

# Create a visual verification grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(8, 9, figsize=(27, 24))
fig.suptitle("V2.0 Seal Verification Grid (72 Demons)", fontsize=18, fontweight='bold')

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

# Clear empty axes
for row_idx in range(8):
    for col_idx in range(9):
        sid = row_idx * 9 + col_idx + 1
        if sid > 72:
            axes[row_idx][col_idx].axis('off')

plt.tight_layout()
plt.savefig(str(OUTDIR / "v2_verification_grid.png"), dpi=150)
plt.close()

print("\nSaved v2_verification_grid.png for visual inspection")
