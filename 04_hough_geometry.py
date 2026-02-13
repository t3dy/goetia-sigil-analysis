"""
Script 4: Hough Line & Circle Detection + Angle Analysis
Detects straight lines, circles/arcs, and analyzes orientation distributions.
"""
import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIGIL_DIR = Path(r"C:\Users\PC\Downloads\goetia_analysis\extracted_sigils")
OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis\hough_results")
OUTDIR.mkdir(parents=True, exist_ok=True)

metadata_file = SIGIL_DIR.parent / "sigil_metadata.json"
with open(metadata_file) as f:
    sigils = json.load(f)

all_angles = []
results = []

for info in sigils[:72]:
    fpath = SIGIL_DIR / info["file"]
    if not fpath.exists():
        continue

    img = cv2.imread(str(fpath))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 50, 150)

    # Hough lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15,
                            minLineLength=15, maxLineGap=5)

    # Hough circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                                minDist=10, param1=100, param2=25,
                                minRadius=3, maxRadius=50)

    vis = img.copy()
    sigil_angles = []

    n_lines = 0
    if lines is not None:
        n_lines = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            sigil_angles.append(angle)
            all_angles.append(angle)
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    n_circles = 0
    if circles is not None:
        circles_rounded = np.uint16(np.around(circles))
        n_circles = len(circles_rounded[0])
        for c in circles_rounded[0]:
            cv2.circle(vis, (c[0], c[1]), c[2], (0, 255, 0), 1)
            cv2.circle(vis, (c[0], c[1]), 2, (0, 255, 0), -1)

    cv2.imwrite(str(OUTDIR / info["file"]), vis)

    # Angle histogram for this sigil
    if sigil_angles:
        hist, _ = np.histogram(sigil_angles, bins=12, range=(0, 180))
        dominant_angle = (np.argmax(hist) * 15 + 7.5)
    else:
        hist = [0] * 12
        dominant_angle = None

    results.append({
        "id": info["id"],
        "n_lines": n_lines,
        "n_circles": n_circles,
        "dominant_angle": round(dominant_angle, 1) if dominant_angle else None,
        "angle_histogram": [int(x) for x in hist],
        "line_circle_ratio": round(n_lines / max(n_circles, 1), 2)
    })

    if info["id"] <= 5:
        print(f"Sigil {info['id']}: {n_lines} lines, {n_circles} circles, "
              f"dominant angle={dominant_angle}")

with open(SIGIL_DIR.parent / "hough_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

# Global angle histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall angle distribution
axes[0].hist(all_angles, bins=36, range=(0, 180), color='steelblue', edgecolor='black')
axes[0].set_xlabel("Line Angle (degrees)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Line Orientations Across All Sigils")
axes[0].axvline(0, color='red', linestyle='--', alpha=0.5, label='Horizontal')
axes[0].axvline(90, color='green', linestyle='--', alpha=0.5, label='Vertical')
axes[0].axvline(45, color='orange', linestyle='--', alpha=0.5, label='45°')
axes[0].axvline(135, color='purple', linestyle='--', alpha=0.5, label='135°')
axes[0].legend()

# Lines vs circles scatter
line_counts = [r["n_lines"] for r in results]
circle_counts = [r["n_circles"] for r in results]
axes[1].scatter(line_counts, circle_counts, alpha=0.6, s=30)
axes[1].set_xlabel("Number of Detected Lines")
axes[1].set_ylabel("Number of Detected Circles")
axes[1].set_title("Lines vs Circles per Sigil")

plt.tight_layout()
plt.savefig(str(SIGIL_DIR.parent / "angle_distribution.png"), dpi=150)
print(f"\nSaved angle distribution plot")
print(f"Total lines detected: {sum(line_counts)}")
print(f"Total circles detected: {sum(circle_counts)}")
