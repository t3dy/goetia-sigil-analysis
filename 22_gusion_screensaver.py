"""
Script 22: Gusion Screensaver — Ambient Analytical Animation
A continuous 20-minute screensaver where 12 oscillating effects simultaneously
transform the Gusion sigil (#11), driven by real analysis parameters.
No title cards, no labels, no segments — just flowing generative art.

Output:
  gusion_screensaver.mp4  (1920x1080, 30fps, ~20 minutes, grayscale)
"""

import cv2
import numpy as np
import sys
import time
import subprocess
import math
from pathlib import Path
from collections import deque
from skimage.morphology import skeletonize

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from imageio_ffmpeg import get_ffmpeg_exe
FFMPEG_PATH = get_ffmpeg_exe()

OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")
WIDTH, HEIGHT = 1920, 1080
FPS = 30
DURATION = 1200.0  # 20 minutes in seconds
TOTAL_FRAMES = int(DURATION * FPS)  # 36000
BG_VALUE = 13
DISPLAY_W = 600

# ============================================================
# PREPROCESSING (reused from script 21)
# ============================================================

class SigilData:
    pass


def cluster_points(points, radius=5):
    if not points:
        return []
    pts = np.array(points, dtype=np.float32)
    used = np.zeros(len(pts), dtype=bool)
    clusters = []
    for i in range(len(pts)):
        if used[i]:
            continue
        dists = np.sqrt(np.sum((pts - pts[i])**2, axis=1))
        group = dists < radius
        used[group] = True
        members = pts[group]
        centroid = members.mean(axis=0)
        clusters.append((int(centroid[0]), int(centroid[1])))
    return clusters


def extract_branches(skel_arr, junctions, endpoints):
    h, w = skel_arr.shape
    junction_set = set(junctions)
    endpoint_set = set(endpoints)
    special = junction_set | endpoint_set
    visited_branches = set()
    branches = []
    starts = list(endpoint_set) + list(junction_set)
    for start in starts:
        sy, sx = start
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < h and 0 <= nx < w and skel_arr[ny, nx]:
                    branch = [(sy, sx)]
                    prev = (sy, sx)
                    curr = (ny, nx)
                    branch.append(curr)
                    while curr not in special or curr == (ny, nx):
                        if curr in special and curr != (ny, nx):
                            break
                        found_next = False
                        for dy2 in [-1, 0, 1]:
                            for dx2 in [-1, 0, 1]:
                                if dy2 == 0 and dx2 == 0:
                                    continue
                                nny, nnx = curr[0] + dy2, curr[1] + dx2
                                if (0 <= nny < h and 0 <= nnx < w and
                                    skel_arr[nny, nnx] and (nny, nnx) != prev and
                                    (nny, nnx) not in set(branch[:-1])):
                                    branch.append((nny, nnx))
                                    prev = curr
                                    curr = (nny, nnx)
                                    found_next = True
                                    break
                            if found_next:
                                break
                        if not found_next:
                            break
                    key = (min(branch[0], branch[-1]), max(branch[0], branch[-1]))
                    if key not in visited_branches and len(branch) > 2:
                        visited_branches.add(key)
                        branches.append(branch)
    return branches


def preprocess():
    print("Preprocessing Gusion sigil data...")
    sd = SigilData()

    raw = cv2.imread(str(OUTDIR / "extracted_sigils" / "sigil_011.png"), cv2.IMREAD_GRAYSCALE)
    h0, w0 = raw.shape
    scale = DISPLAY_W / w0
    sd.display_h = int(h0 * scale)
    sd.display_w = DISPLAY_W
    sd.sigil = cv2.resize(raw, (sd.display_w, sd.display_h), interpolation=cv2.INTER_AREA)
    print(f"  Sigil: {w0}x{h0} -> {sd.display_w}x{sd.display_h}")

    _, sd.binary = cv2.threshold(sd.sigil, 180, 255, cv2.THRESH_BINARY_INV)

    skel_bool = skeletonize(sd.binary > 0)
    sd.skeleton = (skel_bool.astype(np.uint8)) * 255
    sd.flesh = sd.binary.copy()
    sd.flesh[sd.skeleton > 0] = 0
    print(f"  Skeleton pixels: {np.sum(sd.skeleton > 0)}")

    n_labels, sd.comp_labels, sd.comp_stats, sd.comp_centroids = \
        cv2.connectedComponentsWithStats(sd.binary, connectivity=8)
    sd.n_components = n_labels - 1
    sd.comp_masks = []
    for i in range(1, n_labels):
        mask = (sd.comp_labels == i).astype(np.uint8) * 255
        sd.comp_masks.append(mask)
    print(f"  Connected components: {sd.n_components}")

    sd.contours, sd.hierarchy = cv2.findContours(
        sd.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sd.hole_contours = []
    if sd.hierarchy is not None:
        hier = sd.hierarchy[0]
        for i, h in enumerate(hier):
            if h[3] >= 0 and cv2.contourArea(sd.contours[i]) > 5:
                sd.hole_contours.append(sd.contours[i])
    print(f"  Contours: {len(sd.contours)}, Holes: {len(sd.hole_contours)}")

    skel_arr = (sd.skeleton > 0).astype(np.uint8)
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_arr, -1, kernel) * skel_arr
    sd.junction_positions = list(zip(*np.where(neighbor_count >= 3)))
    sd.endpoint_positions = list(zip(*np.where(neighbor_count == 1)))
    sd.junction_clusters = cluster_points(sd.junction_positions, radius=5)
    sd.branches = extract_branches(skel_arr, sd.junction_positions, sd.endpoint_positions)
    # Sort branches by length for cascade effect
    sd.branches_sorted = sorted(sd.branches, key=len)
    print(f"  Junctions: {len(sd.junction_clusters)}, Endpoints: {len(sd.endpoint_positions)}")
    print(f"  Branches: {len(sd.branches)}")

    edges = cv2.Canny(sd.sigil, 50, 150)
    sd.hough_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15,
                                      minLineLength=10, maxLineGap=5)
    if sd.hough_lines is None:
        sd.hough_lines = np.array([]).reshape(0, 1, 4)
    circles = cv2.HoughCircles(sd.sigil, cv2.HOUGH_GRADIENT, dp=1.2,
                                minDist=8, param1=100, param2=20,
                                minRadius=3, maxRadius=50)
    if circles is not None:
        sd.hough_circles = np.round(circles[0]).astype(int)
    else:
        sd.hough_circles = np.array([]).reshape(0, 3)
    print(f"  Hough lines: {len(sd.hough_lines)}, circles: {len(sd.hough_circles)}")

    # Quadrant masks
    sd.quadrant_masks = []
    mid_y, mid_x = sd.display_h // 2, sd.display_w // 2
    for qy, qx in [(0, 0), (0, mid_x), (mid_y, 0), (mid_y, mid_x)]:
        mask = np.zeros_like(sd.binary)
        ey = mid_y if qy == 0 else sd.display_h
        ex = mid_x if qx == 0 else sd.display_w
        mask[qy:ey, qx:ex] = sd.binary[qy:ey, qx:ex]
        sd.quadrant_masks.append(mask)

    # Fourier descriptors for largest contour
    if sd.contours:
        largest = max(sd.contours, key=cv2.contourArea)
        sd.fourier_contour = largest.squeeze()
        if len(sd.fourier_contour.shape) == 2 and sd.fourier_contour.shape[1] == 2:
            z = sd.fourier_contour[:, 0] + 1j * sd.fourier_contour[:, 1]
            sd.fourier_coeffs = np.fft.fft(z)
        else:
            sd.fourier_coeffs = None
    else:
        sd.fourier_contour = None
        sd.fourier_coeffs = None

    # Precompute erosion/dilation sequence (10 levels each direction)
    sd.erosion_cache = []
    sd.dilation_cache = []
    kern3 = np.ones((3, 3), np.uint8)
    img = sd.binary.copy()
    for step in range(10):
        img = cv2.erode(img, kern3, iterations=1)
        sd.erosion_cache.append(img.copy())
    img = sd.binary.copy()
    for step in range(10):
        img = cv2.dilate(img, kern3, iterations=1)
        sd.dilation_cache.append(img.copy())

    # Precompute undulation remap tables (cache a few phases)
    sd.remap_cache = {}

    # Canvas position
    sd.cx = WIDTH // 2
    sd.cy = HEIGHT // 2

    print("Preprocessing complete!\n")
    return sd


# ============================================================
# FRAME HELPERS
# ============================================================

def place_image(frame, img, cx, cy):
    h, w = img.shape[:2]
    x1, y1 = cx - w // 2, cy - h // 2
    sx1, sy1 = max(0, -x1), max(0, -y1)
    sx2 = w - max(0, x1 + w - WIDTH)
    sy2 = h - max(0, y1 + h - HEIGHT)
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(WIDTH, x1 + w), min(HEIGHT, y1 + h)
    if fx2 > fx1 and fy2 > fy1:
        src = img[sy1:sy2, sx1:sx2]
        frame[fy1:fy2, fx1:fx2] = np.maximum(frame[fy1:fy2, fx1:fx2], src)


def place_image_alpha(frame, img, cx, cy, alpha=1.0):
    h, w = img.shape[:2]
    x1, y1 = cx - w // 2, cy - h // 2
    sx1, sy1 = max(0, -x1), max(0, -y1)
    sx2 = w - max(0, x1 + w - WIDTH)
    sy2 = h - max(0, y1 + h - HEIGHT)
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(WIDTH, x1 + w), min(HEIGHT, y1 + h)
    if fx2 > fx1 and fy2 > fy1:
        src = img[sy1:sy2, sx1:sx2].astype(np.float32)
        dst = frame[fy1:fy2, fx1:fx2].astype(np.float32)
        blended = dst * (1 - alpha) + src * alpha
        frame[fy1:fy2, fx1:fx2] = np.clip(blended, 0, 255).astype(np.uint8)


def osc(t, period, phase=0.0):
    """Oscillator: returns value in [0, 1] based on sine wave."""
    return 0.5 + 0.5 * math.sin(2 * math.pi * t / period + phase)


def ease(x):
    """Smooth easing (smoothstep)."""
    x = max(0, min(1, x))
    return x * x * (3 - 2 * x)


# ============================================================
# 12 EFFECT FUNCTIONS
# Each takes (sd, t, intensity) and returns a grayscale image
# the same size as the sigil (sd.display_h x sd.display_w)
# ============================================================

def fx_quadrant_drift(sd, t, intensity):
    """Quadrants separate, rotate, and return. intensity=0: together, 1: max separation."""
    canvas = np.zeros((sd.display_h, sd.display_w), dtype=np.uint8)
    sep = intensity * 40  # max pixel separation
    rotation = intensity * 15  # max degrees rotation

    offsets = [(-1, -1), (1, -1), (-1, 1), (1, 1)]  # TL, TR, BL, BR
    for i, mask in enumerate(sd.quadrant_masks):
        dx = int(offsets[i][0] * sep)
        dy = int(offsets[i][1] * sep)
        rot_deg = rotation * (i + 1) * 0.5 * math.sin(t * 0.03 + i)

        h, w = mask.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), rot_deg, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        warped = cv2.warpAffine(mask, M, (w, h), borderValue=0)
        canvas = np.maximum(canvas, warped)
    return canvas


def fx_skeleton_breathe(sd, t, intensity):
    """Skeleton dilates/erodes. intensity oscillates dilation amount."""
    # Map intensity 0->erode, 0.5->normal, 1->dilate
    level = int((intensity - 0.5) * 6)  # -3 to +3
    kern = np.ones((2, 2), np.uint8)
    if level > 0:
        return cv2.dilate(sd.skeleton, kern, iterations=level)
    elif level < 0:
        return cv2.erode(sd.skeleton, kern, iterations=abs(level))
    return sd.skeleton.copy()


def fx_hole_pulse(sd, t, intensity):
    """Holes fill and empty with phase-shifted pulses."""
    canvas = np.zeros((sd.display_h, sd.display_w), dtype=np.uint8)
    for i, contour in enumerate(sd.hole_contours):
        # Each hole has its own phase
        hole_intensity = osc(t, 20.0, phase=i * math.pi / len(sd.hole_contours) * 2)
        fill_alpha = hole_intensity * intensity
        if fill_alpha > 0.1:
            mask = np.zeros_like(canvas)
            cv2.drawContours(mask, [contour], -1, 255, -1)  # filled
            canvas = np.maximum(canvas, (mask.astype(np.float32) * fill_alpha).astype(np.uint8))
    return canvas


def fx_branch_cascade(sd, t, intensity):
    """Branches appear/disappear by length order."""
    canvas = np.zeros((sd.display_h, sd.display_w), dtype=np.uint8)
    n_branches = len(sd.branches_sorted)
    if n_branches == 0:
        return canvas

    # How many branches visible: cycles through revealing then hiding
    cycle_t = (t % 35.0) / 35.0  # 0 to 1 over 35 seconds
    if cycle_t < 0.5:
        visible_frac = ease(cycle_t * 2) * intensity
    else:
        visible_frac = ease(1.0 - (cycle_t - 0.5) * 2) * intensity

    n_visible = int(visible_frac * n_branches)

    for branch in sd.branches_sorted[:n_visible]:
        for y, x in branch:
            if 0 <= y < sd.display_h and 0 <= x < sd.display_w:
                canvas[y, x] = 255
    # Dilate for visibility
    if n_visible > 0:
        canvas = cv2.dilate(canvas, np.ones((2, 2), np.uint8), iterations=1)
    return canvas


def fx_symmetry_ghost(sd, t, intensity):
    """Mirror overlay fades in and out. Alternates H and V mirrors."""
    # Alternate between H and V mirror every ~25 seconds
    use_h = (int(t / 25) % 2) == 0
    if use_h:
        mirror = cv2.flip(sd.binary, 1)  # horizontal flip
    else:
        mirror = cv2.flip(sd.binary, 0)  # vertical flip

    alpha = intensity * 0.6  # max 60% opacity for the ghost
    blended = (sd.binary.astype(np.float32) * (1 - alpha) +
               mirror.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def fx_line_rain(sd, t, intensity):
    """Hough lines slide across the image at their detected angles."""
    canvas = np.zeros((sd.display_h, sd.display_w), dtype=np.uint8)
    if len(sd.hough_lines) == 0:
        return canvas

    # Only draw a subset of lines, cycling through them
    n_lines = len(sd.hough_lines)
    # Each line slides along its perpendicular direction
    line_brightness = int(intensity * 180)
    if line_brightness < 10:
        return canvas

    # Show up to 30 lines at a time, rotating through the set
    offset = int(t * 2) % n_lines
    count = min(30, n_lines)
    for idx in range(count):
        line_idx = (offset + idx * (n_lines // count)) % n_lines
        x1, y1, x2, y2 = sd.hough_lines[line_idx][0]
        # Slide offset along perpendicular
        angle = math.atan2(y2 - y1, x2 - x1)
        perp_angle = angle + math.pi / 2
        slide = math.sin(t * 0.5 + idx * 0.3) * intensity * 15
        dx = int(slide * math.cos(perp_angle))
        dy = int(slide * math.sin(perp_angle))
        cv2.line(canvas, (x1 + dx, y1 + dy), (x2 + dx, y2 + dy),
                 line_brightness, 1, cv2.LINE_AA)
    return canvas


def fx_circle_ripple(sd, t, intensity):
    """Detected circles expand outward as ripples."""
    canvas = np.zeros((sd.display_h, sd.display_w), dtype=np.uint8)
    if len(sd.hough_circles) == 0:
        return canvas

    brightness = int(intensity * 150)
    if brightness < 10:
        return canvas

    # Show up to 20 circles, pulsing their radii
    count = min(20, len(sd.hough_circles))
    for idx in range(count):
        cx, cy, r = sd.hough_circles[idx % len(sd.hough_circles)]
        # Pulse radius
        pulse = 1.0 + 0.4 * math.sin(t * 0.3 + idx * 0.5) * intensity
        new_r = max(1, int(r * pulse))
        cv2.circle(canvas, (int(cx), int(cy)), new_r, brightness, 1, cv2.LINE_AA)
    return canvas


def fx_fourier_morph(sd, t, intensity):
    """Contour shape oscillates between few and many harmonics."""
    canvas = np.zeros((sd.display_h, sd.display_w), dtype=np.uint8)
    if sd.fourier_coeffs is None:
        return canvas

    n = len(sd.fourier_coeffs)
    # Number of harmonics: oscillates from 2 to n
    min_harm = 2
    max_harm = min(64, n)
    n_harmonics = int(min_harm + (max_harm - min_harm) * intensity)

    # Reconstruct contour with limited harmonics
    coeffs_filtered = np.zeros_like(sd.fourier_coeffs)
    coeffs_filtered[:n_harmonics] = sd.fourier_coeffs[:n_harmonics]
    if n_harmonics < n:
        coeffs_filtered[-n_harmonics:] = sd.fourier_coeffs[-n_harmonics:]

    z_reconstructed = np.fft.ifft(coeffs_filtered)
    pts = np.column_stack([z_reconstructed.real, z_reconstructed.imag]).astype(np.int32)
    pts = pts.reshape(-1, 1, 2)

    brightness = int(80 + 100 * intensity)
    cv2.drawContours(canvas, [pts], -1, brightness, 1, cv2.LINE_AA)
    return canvas


def fx_undulation(sd, t, intensity):
    """Sine-wave spatial distortion."""
    h, w = sd.display_h, sd.display_w
    amplitude = intensity * 8  # max 8 pixel displacement
    if amplitude < 0.5:
        return sd.binary.copy()

    # Build remap tables
    yy, xx = np.mgrid[0:h, 0:w]
    freq = 0.02 + 0.01 * math.sin(t * 0.1)  # slowly varying frequency
    map_x = (xx + amplitude * np.sin(freq * yy + t * 0.5)).astype(np.float32)
    map_y = (yy + amplitude * np.sin(freq * xx + t * 0.3)).astype(np.float32)
    return cv2.remap(sd.binary, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)


def fx_erosion_tide(sd, t, intensity):
    """Morphological erosion/dilation cycle using precomputed cache."""
    # intensity 0 = max erosion, 0.5 = original, 1 = max dilation
    if intensity < 0.5:
        # Erosion: index 0 (lightest) to 9 (heaviest)
        idx = int((0.5 - intensity) * 2 * 9)
        idx = min(idx, len(sd.erosion_cache) - 1)
        return sd.erosion_cache[idx]
    elif intensity > 0.5:
        idx = int((intensity - 0.5) * 2 * 9)
        idx = min(idx, len(sd.dilation_cache) - 1)
        return sd.dilation_cache[idx]
    return sd.binary.copy()


def fx_contour_peel(sd, t, intensity):
    """Outer contours drift away and return."""
    canvas = np.zeros((sd.display_h, sd.display_w), dtype=np.uint8)
    if not sd.contours:
        return canvas

    # Sort contours by area (largest first)
    sorted_contours = sorted(sd.contours, key=cv2.contourArea, reverse=True)
    n_contours = len(sorted_contours)

    for i, contour in enumerate(sorted_contours):
        # Outer contours (low i) drift more, inner contours (high i) drift less
        drift_scale = max(0, 1.0 - i / max(n_contours, 1)) * intensity
        angle = t * 0.1 + i * 2 * math.pi / n_contours
        dx = int(drift_scale * 30 * math.cos(angle))
        dy = int(drift_scale * 30 * math.sin(angle))

        shifted = contour.copy()
        shifted[:, :, 0] += dx
        shifted[:, :, 1] += dy

        brightness = int(180 - i * 15)
        brightness = max(40, min(brightness, 220))
        cv2.drawContours(canvas, [shifted], -1, brightness, 1, cv2.LINE_AA)
    return canvas


def fx_node_sparkle(sd, t, intensity):
    """Junction and endpoint glow pulses."""
    canvas = np.zeros((sd.display_h, sd.display_w), dtype=np.uint8)
    brightness = int(intensity * 220)
    if brightness < 10:
        return canvas

    # Junctions: larger glow
    for i, (jy, jx) in enumerate(sd.junction_clusters):
        glow = osc(t, 3.0, phase=i * 0.7)
        r = int(3 + 4 * glow * intensity)
        b = int(brightness * glow)
        if b > 10:
            cv2.circle(canvas, (jx, jy), r, b, -1, cv2.LINE_AA)

    # Endpoints: smaller, faster
    for i, (ey, ex) in enumerate(sd.endpoint_positions):
        glow = osc(t, 2.0, phase=i * 1.1 + math.pi)
        r = int(2 + 3 * glow * intensity)
        b = int(brightness * glow * 0.7)
        if b > 10:
            cv2.circle(canvas, (ex, ey), r, b, -1, cv2.LINE_AA)
    return canvas


# ============================================================
# OSCILLATOR CONFIGURATION
# ============================================================

# (effect_function, period_seconds, phase_offset, weight)
# Weights control how strongly each effect appears (0.0 to 1.0)
OSCILLATORS = [
    (fx_quadrant_drift,    45.0,  0.0,           0.7),
    (fx_skeleton_breathe,  30.0,  math.pi/6,     0.5),
    (fx_hole_pulse,        20.0,  math.pi/3,     0.4),
    (fx_branch_cascade,    35.0,  math.pi/2,     0.6),
    (fx_symmetry_ghost,    50.0,  2*math.pi/3,   0.35),
    (fx_line_rain,         25.0,  5*math.pi/6,   0.3),
    (fx_circle_ripple,     40.0,  math.pi,       0.3),
    (fx_fourier_morph,     55.0,  7*math.pi/6,   0.4),
    (fx_undulation,        15.0,  4*math.pi/3,   0.6),
    (fx_erosion_tide,      60.0,  3*math.pi/2,   0.5),
    (fx_contour_peel,      50.0,  5*math.pi/3,   0.35),
    (fx_node_sparkle,       8.0,  11*math.pi/6,  0.3),
]

OSCILLATOR_NAMES = [
    "quadrant_drift", "skeleton_breathe", "hole_pulse", "branch_cascade",
    "symmetry_ghost", "line_rain", "circle_ripple", "fourier_morph",
    "undulation", "erosion_tide", "contour_peel", "node_sparkle"
]


# ============================================================
# MAIN RENDER
# ============================================================

def render_frames(sd):
    """Generator yielding 1920x1080 grayscale frames."""
    print(f"Rendering {TOTAL_FRAMES} frames ({DURATION:.0f}s at {FPS}fps)...")

    for frame_idx in range(TOTAL_FRAMES):
        t = frame_idx / FPS  # time in seconds

        # Start with dark canvas
        canvas = np.full((HEIGHT, WIDTH), BG_VALUE, dtype=np.float32)

        # Compute all oscillator intensities
        intensities = []
        for fx_func, period, phase, weight in OSCILLATORS:
            intensity = osc(t, period, phase)
            intensities.append(intensity)

        # --- LAYER 1: Base sigil (always partially visible) ---
        # The base sigil fades in/out slightly based on erosion tide
        erosion_intensity = intensities[9]  # erosion_tide
        base_img = fx_erosion_tide(sd, t, erosion_intensity)

        # Apply undulation to the base
        undulation_intensity = intensities[8]  # undulation
        if undulation_intensity > 0.1:
            base_img = fx_undulation(sd, t, undulation_intensity)

        # Apply quadrant drift to the base when active
        quad_intensity = intensities[0]
        if quad_intensity > 0.3:
            quad_img = fx_quadrant_drift(sd, t, quad_intensity)
            # Blend between normal base and quadrant-drifted version
            blend = ease(quad_intensity - 0.3) / 0.7  # 0 at 0.3, 1 at 1.0
            base_img = np.clip(
                base_img.astype(np.float32) * (1 - blend) +
                quad_img.astype(np.float32) * blend,
                0, 255
            ).astype(np.uint8)

        # Place base image
        h, w = base_img.shape
        x1 = sd.cx - w // 2
        y1 = sd.cy - h // 2
        fx1, fy1 = max(0, x1), max(0, y1)
        fx2, fy2 = min(WIDTH, x1 + w), min(HEIGHT, y1 + h)
        sx1, sy1 = max(0, -x1), max(0, -y1)
        if fx2 > fx1 and fy2 > fy1:
            src = base_img[sy1:sy1 + (fy2 - fy1), sx1:sx1 + (fx2 - fx1)]
            canvas[fy1:fy2, fx1:fx2] = np.maximum(
                canvas[fy1:fy2, fx1:fx2], src.astype(np.float32))

        # --- LAYER 2: Overlay effects ---
        overlay_effects = [
            (fx_skeleton_breathe, 1, 0.5),   # skeleton breathe
            (fx_hole_pulse,       2, 0.4),   # hole pulse
            (fx_branch_cascade,   3, 0.6),   # branch cascade
            (fx_symmetry_ghost,   4, 0.35),  # symmetry ghost
            (fx_line_rain,        5, 0.3),   # line rain
            (fx_circle_ripple,    6, 0.3),   # circle ripple
            (fx_fourier_morph,    7, 0.4),   # fourier morph
            (fx_contour_peel,     10, 0.35), # contour peel
            (fx_node_sparkle,     11, 0.3),  # node sparkle
        ]

        for fx_func, osc_idx, weight in overlay_effects:
            intensity = intensities[osc_idx]
            if intensity * weight < 0.05:
                continue  # skip nearly invisible effects

            layer = fx_func(sd, t, intensity)
            alpha = intensity * weight

            # Place layer centered on canvas
            h, w = layer.shape
            x1 = sd.cx - w // 2
            y1 = sd.cy - h // 2
            fx1, fy1 = max(0, x1), max(0, y1)
            fx2, fy2 = min(WIDTH, x1 + w), min(HEIGHT, y1 + h)
            sx1, sy1 = max(0, -x1), max(0, -y1)
            if fx2 > fx1 and fy2 > fy1:
                src = layer[sy1:sy1 + (fy2 - fy1), sx1:sx1 + (fx2 - fx1)].astype(np.float32)
                region = canvas[fy1:fy2, fx1:fx2]
                canvas[fy1:fy2, fx1:fx2] = np.maximum(region, src * alpha)

        # Convert to uint8
        frame = np.clip(canvas, 0, 255).astype(np.uint8)
        yield frame

        # Progress reporting every 900 frames (30 seconds of video)
        if (frame_idx + 1) % 900 == 0:
            elapsed_video = (frame_idx + 1) / FPS
            print(f"  {elapsed_video:.0f}s / {DURATION:.0f}s rendered "
                  f"({(frame_idx + 1) / TOTAL_FRAMES * 100:.1f}%)")


def main():
    print("=" * 60)
    print("GUSION SCREENSAVER: AMBIENT ANALYTICAL ANIMATION")
    print("=" * 60)

    sd = preprocess()

    output_path = str(OUTDIR / "gusion_screensaver.mp4")

    total_frames = 0
    start_time = time.time()

    # Start ffmpeg pipe
    ffmpeg_cmd = [
        FFMPEG_PATH, '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{WIDTH}x{HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(FPS), '-i', '-',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        output_path
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        for frame in render_frames(sd):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            proc.stdin.write(frame_bgr.tobytes())
            total_frames += 1

        proc.stdin.close()
        print(f"\nWaiting for ffmpeg to finalize MP4...")
        stdout, stderr = proc.communicate(timeout=600)

        if proc.returncode != 0:
            print(f"ffmpeg error: {stderr.decode('utf-8', errors='replace')[-500:]}")
        else:
            print(f"ffmpeg finished successfully")

    except Exception as e:
        print(f"\nError during rendering: {e}")
        import traceback
        traceback.print_exc()
        proc.stdin.close()
        proc.wait(timeout=30)
        raise

    render_time = time.time() - start_time
    total_duration = total_frames / FPS

    print(f"\n{'=' * 60}")
    print(f"COMPLETE!")
    print(f"  Output: {output_path}")
    print(f"  Frames: {total_frames}")
    print(f"  Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"  Total time: {render_time:.0f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
