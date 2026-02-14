"""
Script 16: Verify Seal-to-Name Mapping
Reads the original image and examines each extracted region to check
which demon name label appears above it. Uses the known grid structure
(the image has text labels like "1. Bael" above each seal).

Strategy: For each extracted sigil bounding box, look at the strip of pixels
just ABOVE the bounding box to find text. Use the x-position within the row
to determine the column index, and use the row to determine the row index.
Then map (row, col) to the traditional Goetia ordering.
"""
import cv2
import numpy as np
from pathlib import Path
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")
INPUT = Path(r"C:\Users\PC\Downloads\1280px-72_Goeta_sigils.png")

# The correct Goetia ordering (1-72)
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

with open(OUTDIR / "sigil_metadata.json") as f:
    sigils = json.load(f)

img = cv2.imread(str(INPUT), cv2.IMREAD_GRAYSCALE)
h, w = img.shape
print(f"Original image size: {w}x{h}")
print(f"Extracted regions: {len(sigils)}")

# ============================================================
# Analyze the grid structure
# ============================================================

# The image has 72 sigils arranged in rows of ~10
# with text labels above each one.
# The source image from Wikipedia shows:
# Row 1: #1-9 (Bael through Paimon), some have "9b. Paimon" as variant
# Row 2: #10-18 (Buer through Bathin)
# etc.

# Let's look at each extracted region and figure out where it falls
# by examining its x,y position in the image grid

# First, cluster by y-coordinate to find rows
bboxes = [(s["id"], s["bbox"][0], s["bbox"][1], s["bbox"][2], s["bbox"][3]) for s in sigils]
# bbox = [x, y, w, h]

# Sort by y to find rows
y_centers = [(sid, y + bh//2) for sid, x, y, bw, bh in bboxes]
y_centers.sort(key=lambda t: t[1])

# Cluster into rows (40px gap threshold)
rows = []
current_row = [y_centers[0]]
for i in range(1, len(y_centers)):
    if y_centers[i][1] - y_centers[i-1][1] > 40:
        rows.append(current_row)
        current_row = [y_centers[i]]
    else:
        current_row.append(y_centers[i])
rows.append(current_row)

print(f"\nFound {len(rows)} rows:")
for i, row in enumerate(rows):
    # Sort row by x position
    row_bboxes = []
    for sid, _ in row:
        for s in sigils:
            if s["id"] == sid:
                row_bboxes.append((s["id"], s["bbox"][0], s["bbox"][1], s["bbox"][2], s["bbox"][3]))
    row_bboxes.sort(key=lambda t: t[1])  # sort by x
    ids_in_row = [rb[0] for rb in row_bboxes]
    xs_in_row = [rb[1] for rb in row_bboxes]
    ws_in_row = [rb[3] for rb in row_bboxes]

    print(f"  Row {i+1}: {len(row)} items, IDs: {ids_in_row}")
    for rb in row_bboxes:
        sid, x, y, bw, bh = rb
        aspect = bw/bh if bh > 0 else 0
        flag = " ***WIDE***" if aspect > 2.0 else ""
        print(f"    #{sid:3d} at ({x:4d},{y:4d}) size {bw:3d}x{bh:3d} aspect={aspect:.2f}{flag}")

# ============================================================
# Now look at the ACTUAL image content to identify problems
# ============================================================

print("\n=== Identifying Problematic Regions ===")
print("Regions with aspect ratio > 2.0 (likely merged seals):")
for s in sigils:
    ar = s["bbox"][2] / s["bbox"][3] if s["bbox"][3] > 0 else 0
    if ar > 2.0:
        print(f"  #{s['id']}: aspect={ar:.2f}, bbox=({s['bbox'][0]},{s['bbox'][1]},{s['bbox'][2]},{s['bbox'][3]})")

# ============================================================
# Determine correct mapping
# ============================================================
# The image layout from inspection:
# Row 1 (y~40-140):  1.Bael  2.Agares  3.Vassago  4.Samigina  5.Marbas  6.Valefor  7.Amon  8.Barbatos  9.Paimon  [9b.Paimon variant]
# Row 2 (y~150-260): 10.Buer 11.Gusion 12.Sitri   13.Beleth   14.Leraje 15.Eligos  16.Zepar 17.Botis   18.Bathin [extra?]
# Row 3 (y~260-360): 19.Sallos 20.Purson 21.Marax  22.Ipos   23.Aim    24.Naberius 25.Glasya 26.Bune  27.Ronove 28.Berith
# Row 4 (y~360-480): 29.Astaroth 30.Forneus 31.Foras 32.Asmoday 33.Gaap 34.Furfur 35.Marchosias 36.Stolas 37.Phenex 38.Halphas
# Row 5 (y~480-600): 39.Malphas 40.Raum 41.Focalor 42.Vepar 43.Sabnock 44.Shax 45.Vine 46.Bifrons 47.Uvall 48.Haagenti
# Row 6 (y~600-720): 49.Crocell 50.Furcas 51.Balam 52.Alloces 53.Camio 54.Murmur 55.Orobas 56.Gremory 57.Ose 58.Amy
# Row 7 (y~720-830): 59.Oriax 60.Vapula 61.Zagan 62.Volac 63.Andras 64.Haures 65.Andrealphus 66.Cimejes 67.Amdusias 68.Belial
# Row 8 (y~830-940): 69.Decarabia 70.Seere 71.Dantalion 72.Andromalius + possible empty/extra regions

# The CORRECT mapping depends on:
# 1. Is there a "9b" variant taking up a slot?
# 2. Did Zepar's wide aspect merge 3 seals into 1?

print("\n=== Building Correct Mapping ===")
print("The image has 10 columns, 8 rows = 80 slots")
print("Row 1 has 9 demons + possibly 9b Paimon variant = 10 slots")
print("Rows 2-7 have 10 demons each = 60 slots")
print("Row 8 has 4 demons + possibly extra = varies")
