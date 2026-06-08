"""
Script 23: Adaptive Preprocessing & Rotation Normalization

Replaces the fixed-threshold binarization (180) used in scripts 01-22.
Steps per sigil:
  1. CLAHE contrast normalization
  2. Otsu's method for binarization (threshold computed per image)
  3. Morphological cleanup (open to remove speckle)
  4. PCA of ink pixel coordinates → rotate so principal axis aligns vertical
  5. Crop to ink bounding box with fixed padding
  6. Quality metrics: otsu_threshold, ink_ratio, contrast_score, rotation_angle

Outputs:
  - normalized_sigils/  (PNG, 256x256, white bg, ink black)
  - preprocessing_quality.json
"""

import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIGIL_DIR = Path(r"C:\Dev\GoetiaRevEng\extracted_sigils_old")
OUTDIR    = Path(r"C:\Dev\GoetiaRevEng")
NORM_DIR  = OUTDIR / "normalized_sigils"
NORM_DIR.mkdir(exist_ok=True)

TARGET_SIZE = 256
PADDING     = 16   # px of white space around ink after crop


def clahe_enhance(gray):
    """Apply CLAHE to boost local contrast before thresholding."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def otsu_binarize(gray):
    """Return binary (ink=255) image and the Otsu threshold used."""
    thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary, int(thresh)


def pca_rotation_angle(binary, elongation_threshold=1.5):
    """
    Compute the angle (degrees) to rotate the image so its principal
    ink axis aligns vertically.  Only applies rotation when the ink
    distribution is genuinely elongated (eigenvalue ratio >= threshold).
    Returns 0 for circular/isotropic distributions.
    """
    ys, xs = np.where(binary > 0)
    if len(ys) < 10:
        return 0.0

    pts = np.stack([xs.astype(float), ys.astype(float)], axis=1)
    pts -= pts.mean(axis=0)

    cov = np.cov(pts.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Only rotate if the distribution is clearly elongated
    ratio = eigenvalues.max() / max(eigenvalues.min(), 1e-6)
    if ratio < elongation_threshold:
        return 0.0

    # Principal axis = eigenvector with largest eigenvalue
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    angle_rad = np.arctan2(principal[1], principal[0])
    # We want the axis vertical (90 deg), so rotate by (90 - angle)
    correction_deg = np.degrees(angle_rad) - 90.0
    # Keep rotation in (-90, +90) range
    while correction_deg > 90:
        correction_deg -= 180
    while correction_deg < -90:
        correction_deg += 180
    return float(correction_deg)


def rotate_image(img, angle_deg):
    """Rotate image around its center, white background."""
    h, w = img.shape
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    # Expand canvas so corners don't clip
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
    return rotated


def crop_to_ink(binary, padding=PADDING):
    """Crop to bounding box of all ink, add fixed padding."""
    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return binary
    y0 = max(0, ys.min() - padding)
    y1 = min(binary.shape[0], ys.max() + padding + 1)
    x0 = max(0, xs.min() - padding)
    x1 = min(binary.shape[1], xs.max() + padding + 1)
    return binary[y0:y1, x0:x1]


def resize_square(binary, size=TARGET_SIZE):
    """Fit the image into a square canvas with white background."""
    canvas = np.zeros((size, size), dtype=np.uint8)
    h, w = binary.shape
    scale = min((size - 2 * PADDING) / max(h, 1), (size - 2 * PADDING) / max(w, 1))
    nh = max(1, int(h * scale))
    nw = max(1, int(w * scale))
    resized = cv2.resize(binary, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def contrast_score(gray):
    """RMS contrast of the grayscale image (0–1)."""
    f = gray.astype(float) / 255.0
    return float(np.sqrt(np.mean((f - f.mean()) ** 2)))


# ── Main loop ──────────────────────────────────────────────────────────────────

sigil_files = sorted(SIGIL_DIR.glob("*.png"))
print(f"Found {len(sigil_files)} sigil images in {SIGIL_DIR}")

quality_records = []

for fpath in sigil_files:
    sid_str = fpath.stem  # e.g. "01", "42"
    try:
        sid = int(sid_str)
    except ValueError:
        sid = sid_str

    gray = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"  WARN: could not read {fpath.name}")
        continue

    orig_h, orig_w = gray.shape
    c_score = contrast_score(gray)

    # 1. CLAHE
    enhanced = clahe_enhance(gray)

    # 2. Otsu binarize
    binary, otsu_t = otsu_binarize(enhanced)

    # 3. Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    ink_ratio_raw = float(np.mean(binary > 0))

    # 4. PCA rotation
    angle = pca_rotation_angle(binary)
    if abs(angle) > 1.0:  # only rotate if non-trivial
        binary = rotate_image(binary, angle)
        binary = (binary > 127).astype(np.uint8) * 255

    # 5. Crop + resize
    binary = crop_to_ink(binary)
    normalized = resize_square(binary)

    ink_ratio_norm = float(np.mean(normalized > 0))

    # Save
    out_path = NORM_DIR / f"{sid_str}.png"
    cv2.imwrite(str(out_path), normalized)

    quality_records.append({
        "id":            sid,
        "file":          fpath.name,
        "orig_size":     [orig_w, orig_h],
        "otsu_threshold": otsu_t,
        "contrast_score": round(c_score, 4),
        "rotation_deg":  round(angle, 2),
        "ink_ratio_raw": round(ink_ratio_raw, 4),
        "ink_ratio_norm": round(ink_ratio_norm, 4),
    })

quality_records.sort(key=lambda r: r["id"] if isinstance(r["id"], int) else 0)

with open(OUTDIR / "preprocessing_quality.json", "w") as f:
    json.dump(quality_records, f, indent=2)

print(f"Normalized {len(quality_records)} sigils -> {NORM_DIR}")

# ── Summary plots ──────────────────────────────────────────────────────────────

thresholds  = [r["otsu_threshold"]  for r in quality_records]
contrasts   = [r["contrast_score"]  for r in quality_records]
rotations   = [r["rotation_deg"]    for r in quality_records]
ink_ratios  = [r["ink_ratio_norm"]  for r in quality_records]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0,0].hist(thresholds, bins=20, color='steelblue', edgecolor='black')
axes[0,0].set_title("Otsu Thresholds (per-image, adaptive)")
axes[0,0].set_xlabel("Threshold value")
axes[0,0].axvline(180, color='red', linestyle='--', label='old fixed=180')
axes[0,0].legend(fontsize=8)

axes[0,1].hist(contrasts, bins=20, color='mediumseagreen', edgecolor='black')
axes[0,1].set_title("Contrast Score (RMS, post-CLAHE)")
axes[0,1].set_xlabel("RMS contrast")

axes[1,0].hist(rotations, bins=20, color='darkorange', edgecolor='black')
axes[1,0].set_title("PCA Rotation Applied (degrees)")
axes[1,0].set_xlabel("Correction angle")
axes[1,0].axvline(0, color='gray', linestyle='--')

axes[1,1].scatter(contrasts, ink_ratios, alpha=0.6, s=40, c='purple')
axes[1,1].set_xlabel("Contrast Score")
axes[1,1].set_ylabel("Ink Ratio (normalized)")
axes[1,1].set_title("Contrast vs Ink Coverage")

plt.suptitle("Preprocessing Quality Report — 23_adaptive_preprocessing.py", y=1.01)
plt.tight_layout()
plt.savefig(str(OUTDIR / "preprocessing_quality.png"), dpi=150, bbox_inches='tight')
print("Saved preprocessing_quality.png")

# Flag potential problem images
low_contrast  = [r for r in quality_records if r["contrast_score"] < 0.05]
extreme_ink   = [r for r in quality_records if r["ink_ratio_norm"] > 0.5 or r["ink_ratio_norm"] < 0.01]
large_rotation = [r for r in quality_records if abs(r["rotation_deg"]) > 45]

print("\n=== Quality Flags ===")
if low_contrast:
    print(f"Low contrast (<0.05): {[r['id'] for r in low_contrast]}")
if extreme_ink:
    print(f"Extreme ink ratio: {[(r['id'], r['ink_ratio_norm']) for r in extreme_ink]}")
if large_rotation:
    print(f"Large rotation applied (>45°): {[(r['id'], r['rotation_deg']) for r in large_rotation]}")
if not (low_contrast or extreme_ink or large_rotation):
    print("No flagged images.")

print(f"\nOtsu threshold range: {min(thresholds)}–{max(thresholds)} (median {int(np.median(thresholds))})")
print(f"Fixed threshold was: 180 — {'matches median' if abs(np.median(thresholds)-180) < 20 else 'differs significantly from median'}")
