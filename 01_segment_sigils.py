"""
Script 1: Grid-based Segmentation of 72 Goetic Sigils
Uses projection profiles to find the grid, then extracts individual sigils.
"""
import cv2
import numpy as np
from pathlib import Path
import json

INPUT = r"C:\Users\PC\Downloads\1280px-72_Goeta_sigils.png"
OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis\extracted_sigils")
OUTDIR.mkdir(parents=True, exist_ok=True)

img = cv2.imread(INPUT)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
print(f"Image size: {w}x{h}")

# Binarize
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Use a smaller dilation to connect strokes within a sigil but not across sigils
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=3)

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get bounding boxes, filter by reasonable size
min_size = 30  # minimum width and height
bboxes_raw = []
for cnt in contours:
    x, y, bw, bh = cv2.boundingRect(cnt)
    if bw > min_size and bh > min_size:
        bboxes_raw.append((x, y, bw, bh))

print(f"After dilation iter=3: {len(bboxes_raw)} regions")

# If too few, many sigils are merging. If too many, noise.
# Let's try different dilation levels to find the sweet spot near 72-80 regions
best_bboxes = bboxes_raw
best_diff = abs(len(bboxes_raw) - 76)  # ~72 sigils + some variant labels

for iters in range(1, 8):
    d = cv2.dilate(binary, kernel, iterations=iters)
    cnts, _ = cv2.findContours(d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in cnts:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw > min_size and bh > min_size:
            boxes.append((x, y, bw, bh))
    diff = abs(len(boxes) - 76)
    print(f"  dilation iters={iters}: {len(boxes)} regions (diff from 76: {diff})")
    if diff < best_diff:
        best_diff = diff
        best_bboxes = boxes

bboxes = best_bboxes
print(f"\nUsing segmentation with {len(bboxes)} regions")

# Sort by position (top-to-bottom, left-to-right)
# Group into rows first
bboxes.sort(key=lambda b: b[1])  # sort by y

# Cluster into rows by y-coordinate
rows = []
current_row = [bboxes[0]]
for bb in bboxes[1:]:
    if bb[1] - current_row[-1][1] > 40:  # new row
        rows.append(current_row)
        current_row = [bb]
    else:
        current_row.append(bb)
rows.append(current_row)

# Sort each row by x
ordered_bboxes = []
for row in rows:
    row.sort(key=lambda b: b[0])
    ordered_bboxes.extend(row)

print(f"Organized into {len(rows)} rows: {[len(r) for r in rows]}")

# Draw bounding boxes on a copy for visualization
vis = img.copy()
sigil_info = []
padding = 5

for i, (x, y, bw, bh) in enumerate(ordered_bboxes):
    cv2.rectangle(vis, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
    cv2.putText(vis, str(i+1), (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Extract with padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)

    sigil = img[y1:y2, x1:x2]
    fname = f"sigil_{i+1:03d}.png"
    cv2.imwrite(str(OUTDIR / fname), sigil)
    sigil_info.append({
        "id": i + 1,
        "bbox": [int(x), int(y), int(bw), int(bh)],
        "file": fname,
        "aspect_ratio": round(bw / bh, 3)
    })

cv2.imwrite(str(OUTDIR.parent / "segmentation_vis.png"), vis)

with open(OUTDIR.parent / "sigil_metadata.json", "w") as f:
    json.dump(sigil_info, f, indent=2)

print(f"\nExtracted {len(sigil_info)} sigils to {OUTDIR}")
if sigil_info:
    ars = [s['aspect_ratio'] for s in sigil_info]
    print(f"Aspect ratio stats: min={min(ars):.2f}, max={max(ars):.2f}, mean={np.mean(ars):.2f}")
