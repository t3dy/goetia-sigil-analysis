"""
Script 21: Gusion Deep-Dive — 20-Minute Analytical Animation
A single-sigil meditation on Gusion (#11) exploring every analysis parameter
as a visual transformation tool. Black and white only, no zoom.

Each 1-minute segment decomposes or transforms the sigil according to a
different measurement from our analysis pipeline, answering the question:
"What does this measurement LOOK like?"

Output:
  gusion_deep_dive.mp4  (1920x1080, 30fps, ~20 minutes)
"""

import cv2
import numpy as np
import json
import sys
import time
import base64
import subprocess
import tempfile
import os
from pathlib import Path
from collections import deque
from skimage.morphology import skeletonize

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from imageio_ffmpeg import get_ffmpeg_exe
FFMPEG_PATH = get_ffmpeg_exe()

OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")
WIDTH, HEIGHT = 1920, 1080
FPS = 30
SEGMENT_DURATION = 60.0  # seconds per segment
TITLE_CARD_DURATION = 3.0  # seconds for title card at start of each segment
BG_VALUE = 13  # dark background (matches dashboard)
DISPLAY_W = 600  # sigil display width on canvas

# ============================================================
# PREPROCESSING
# ============================================================

class SigilData:
    """All preprocessed data for the Gusion sigil."""
    pass


def preprocess():
    """Load and preprocess everything needed for all 20 segments."""
    print("Preprocessing Gusion sigil data...")
    sd = SigilData()

    # 1. Load sigil and upscale
    raw = cv2.imread(str(OUTDIR / "extracted_sigils" / "sigil_011.png"), cv2.IMREAD_GRAYSCALE)
    h0, w0 = raw.shape
    scale = DISPLAY_W / w0
    sd.display_h = int(h0 * scale)
    sd.display_w = DISPLAY_W
    sd.sigil = cv2.resize(raw, (sd.display_w, sd.display_h), interpolation=cv2.INTER_AREA)
    print(f"  Sigil: {w0}x{h0} -> {sd.display_w}x{sd.display_h}")

    # 2. Binary version
    _, sd.binary = cv2.threshold(sd.sigil, 180, 255, cv2.THRESH_BINARY_INV)

    # 3. Skeleton
    skel_bool = skeletonize(sd.binary > 0)
    sd.skeleton = (skel_bool.astype(np.uint8)) * 255
    print(f"  Skeleton pixels: {np.sum(sd.skeleton > 0)}")

    # 4. Flesh (sigil minus skeleton)
    sd.flesh = sd.binary.copy()
    sd.flesh[sd.skeleton > 0] = 0

    # 5. Connected components
    n_labels, sd.comp_labels, sd.comp_stats, sd.comp_centroids = \
        cv2.connectedComponentsWithStats(sd.binary, connectivity=8)
    sd.n_components = n_labels - 1  # exclude background
    print(f"  Connected components: {sd.n_components}")

    # Create component masks (list of binary masks, one per component)
    sd.comp_masks = []
    for i in range(1, n_labels):
        mask = (sd.comp_labels == i).astype(np.uint8) * 255
        sd.comp_masks.append(mask)

    # 6. Contours with hierarchy
    sd.contours, sd.hierarchy = cv2.findContours(
        sd.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Contours: {len(sd.contours)}")

    # Find hole contours (contours with a parent)
    sd.hole_contours = []
    if sd.hierarchy is not None:
        hier = sd.hierarchy[0]
        for i, h in enumerate(hier):
            parent = h[3]
            if parent >= 0:  # has a parent = could be a hole
                # Check if this contour is actually enclosed
                area = cv2.contourArea(sd.contours[i])
                if area > 5:  # skip tiny noise
                    sd.hole_contours.append(sd.contours[i])
    print(f"  Hole contours: {len(sd.hole_contours)}")

    # 7. Junction and endpoint positions from skeleton
    skel_arr = (sd.skeleton > 0).astype(np.uint8)
    # Count neighbors for each skeleton pixel
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_arr, -1, kernel)
    neighbor_count = neighbor_count * skel_arr  # only count for skeleton pixels

    sd.junction_positions = list(zip(*np.where(neighbor_count >= 3)))  # (y, x) tuples
    sd.endpoint_positions = list(zip(*np.where(neighbor_count == 1)))
    print(f"  Raw junctions: {len(sd.junction_positions)}, endpoints: {len(sd.endpoint_positions)}")

    # Cluster junction positions (nearby junctions are the same logical junction)
    sd.junction_clusters = cluster_points(sd.junction_positions, radius=5)
    print(f"  Junction clusters: {len(sd.junction_clusters)}")

    # 8. Build skeleton graph (branches as paths between junctions/endpoints)
    sd.branches = extract_branches(skel_arr, sd.junction_positions, sd.endpoint_positions)
    print(f"  Skeleton branches: {len(sd.branches)}")

    # 9. BFS ordering of skeleton pixels for tracing
    sd.trace_order = compute_trace_order(skel_arr, sd.endpoint_positions)
    print(f"  Trace order: {len(sd.trace_order)} pixels")

    # 10. Hough lines and circles (re-detect for coordinates)
    edges = cv2.Canny(sd.sigil, 50, 150)
    sd.hough_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15,
                                      minLineLength=10, maxLineGap=5)
    if sd.hough_lines is None:
        sd.hough_lines = np.array([]).reshape(0, 1, 4)
    else:
        sd.hough_lines = sd.hough_lines
    print(f"  Hough lines: {len(sd.hough_lines)}")

    circles = cv2.HoughCircles(sd.sigil, cv2.HOUGH_GRADIENT, dp=1.2,
                                minDist=8, param1=100, param2=20,
                                minRadius=3, maxRadius=50)
    if circles is not None:
        sd.hough_circles = np.round(circles[0]).astype(int)
    else:
        sd.hough_circles = np.array([]).reshape(0, 3)
    print(f"  Hough circles: {len(sd.hough_circles)}")

    # 11. Gradient orientations for angle decomposition
    gx = cv2.Sobel(sd.sigil, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(sd.sigil, cv2.CV_64F, 0, 1, ksize=3)
    sd.gradient_angle = np.arctan2(gy, gx) * 180 / np.pi  # -180 to 180
    sd.gradient_angle = sd.gradient_angle % 180  # 0 to 180
    sd.gradient_magnitude = np.sqrt(gx**2 + gy**2)

    # Create angle bin masks (12 bins of 15 degrees each)
    sd.angle_masks = []
    for b in range(12):
        angle_min = b * 15
        angle_max = (b + 1) * 15
        mask = ((sd.gradient_angle >= angle_min) & (sd.gradient_angle < angle_max) &
                (sd.gradient_magnitude > 20) & (sd.binary > 0))
        sd.angle_masks.append(mask.astype(np.uint8) * 255)

    # 12. Radial ring masks
    cy, cx = sd.display_h / 2, sd.display_w / 2
    yy, xx = np.mgrid[0:sd.display_h, 0:sd.display_w]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    max_dist = max(cx, cy)
    sd.radial_masks = []
    for ring in range(8):
        r_inner = ring / 8 * max_dist
        r_outer = (ring + 1) / 8 * max_dist
        mask = ((dist >= r_inner) & (dist < r_outer) & (sd.binary > 0))
        sd.radial_masks.append(mask.astype(np.uint8) * 255)

    # 13. Quadrant masks
    sd.quadrant_masks = []
    mid_y, mid_x = sd.display_h // 2, sd.display_w // 2
    for qy, qx in [(0, 0), (0, mid_x), (mid_y, 0), (mid_y, mid_x)]:
        mask = np.zeros_like(sd.binary)
        ey = mid_y if qy == 0 else sd.display_h
        ex = mid_x if qx == 0 else sd.display_w
        mask[qy:ey, qx:ex] = sd.binary[qy:ey, qx:ex]
        sd.quadrant_masks.append(mask)

    # 14. Fourier descriptors for largest contour
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

    # Canvas position (centered)
    sd.cx = WIDTH // 2
    sd.cy = HEIGHT // 2

    print("Preprocessing complete!\n")
    return sd


def cluster_points(points, radius=5):
    """Cluster nearby points into groups, return list of (cy, cx) centroids."""
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
    """Extract skeleton branches as lists of (y,x) pixel coordinates."""
    h, w = skel_arr.shape
    # Mark junction and endpoint pixels
    junction_set = set(junctions)
    endpoint_set = set(endpoints)
    special = junction_set | endpoint_set

    visited_branches = set()
    branches = []

    # Start from each endpoint and junction, trace along the skeleton
    starts = list(endpoint_set) + list(junction_set)

    for start in starts:
        sy, sx = start
        # Find neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < h and 0 <= nx < w and skel_arr[ny, nx]:
                    # Trace this branch
                    branch = [(sy, sx)]
                    prev = (sy, sx)
                    curr = (ny, nx)
                    branch.append(curr)

                    while curr not in special or curr == (ny, nx):
                        if curr in special and curr != (ny, nx):
                            break
                        # Find next pixel
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

                    # Deduplicate: use frozenset of first and last point
                    key = (min(branch[0], branch[-1]), max(branch[0], branch[-1]))
                    if key not in visited_branches and len(branch) > 2:
                        visited_branches.add(key)
                        branches.append(branch)

    return branches


def compute_trace_order(skel_arr, endpoints):
    """BFS from an endpoint to get pixel reveal order."""
    h, w = skel_arr.shape
    if not endpoints:
        # Fallback: any skeleton pixel
        ys, xs = np.where(skel_arr > 0)
        if len(ys) == 0:
            return []
        start = (ys[0], xs[0])
    else:
        start = endpoints[0]

    visited = set()
    order = []
    queue = deque([start])
    visited.add(start)

    while queue:
        y, x = queue.popleft()
        order.append((y, x))
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (0 <= ny < h and 0 <= nx < w and
                    skel_arr[ny, nx] > 0 and (ny, nx) not in visited):
                    visited.add((ny, nx))
                    queue.append((ny, nx))

    return order


# ============================================================
# FRAME HELPERS
# ============================================================

def make_frame(sd):
    """Create a blank 1080p frame."""
    return np.full((HEIGHT, WIDTH), BG_VALUE, dtype=np.uint8)


def place_image(frame, img, cx, cy):
    """Place a grayscale image centered at (cx, cy) on a grayscale frame."""
    h, w = img.shape[:2]
    x1 = cx - w // 2
    y1 = cy - h // 2

    # Clip regions
    sx1, sy1 = max(0, -x1), max(0, -y1)
    sx2, sy2 = w - max(0, x1 + w - WIDTH), h - max(0, y1 + h - HEIGHT)
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(WIDTH, x1 + w), min(HEIGHT, y1 + h)

    if fx2 > fx1 and fy2 > fy1:
        src = img[sy1:sy2, sx1:sx2]
        frame[fy1:fy2, fx1:fx2] = np.maximum(frame[fy1:fy2, fx1:fx2], src)


def place_image_alpha(frame, img, cx, cy, alpha=1.0):
    """Place image with alpha blending."""
    h, w = img.shape[:2]
    x1 = cx - w // 2
    y1 = cy - h // 2

    sx1, sy1 = max(0, -x1), max(0, -y1)
    sx2, sy2 = w - max(0, x1 + w - WIDTH), h - max(0, y1 + h - HEIGHT)
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(WIDTH, x1 + w), min(HEIGHT, y1 + h)

    if fx2 > fx1 and fy2 > fy1:
        src = img[sy1:sy2, sx1:sx2].astype(np.float32)
        dst = frame[fy1:fy2, fx1:fx2].astype(np.float32)
        blended = dst * (1 - alpha) + src * alpha
        frame[fy1:fy2, fx1:fx2] = np.clip(blended, 0, 255).astype(np.uint8)


def put_text_gray(frame, text, pos, font_scale=1.0, color=230, thickness=2,
                  font=cv2.FONT_HERSHEY_SIMPLEX, center=False):
    """Draw white text on grayscale frame."""
    if center:
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        pos = (pos[0] - tw // 2, pos[1] + th // 2)
    cv2.putText(frame, text, pos, font, font_scale, int(color), thickness, cv2.LINE_AA)


def title_card(title, subtitle=""):
    """Generate title card frames for segment start (generator)."""
    n = int(TITLE_CARD_DURATION * FPS)
    for i in range(n):
        t = i / FPS
        alpha = min(1.0, t / 0.5) * min(1.0, (TITLE_CARD_DURATION - t) / 0.5)
        frame = make_frame(None)
        color = int(230 * alpha)
        put_text_gray(frame, title, (WIDTH // 2, HEIGHT // 2 - 20),
                      1.5, color, 3, center=True)
        if subtitle:
            sub_color = int(140 * alpha)
            put_text_gray(frame, subtitle, (WIDTH // 2, HEIGHT // 2 + 30),
                          0.7, sub_color, 1, center=True)
        yield frame


def lerp(a, b, t):
    """Linear interpolation."""
    return a + (b - a) * t


def ease_in_out(t):
    """Smooth ease in/out curve."""
    return t * t * (3 - 2 * t)


# ============================================================
# SEGMENT IMPLEMENTATIONS
# ============================================================

def segment_01_skeleton_trace(sd):
    """The Whole — pixel-by-pixel skeleton trace."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("The Whole", "Pixel-by-pixel skeleton trace")

    trace = sd.trace_order
    total_pixels = len(trace)
    pixels_per_frame = max(1, total_pixels // n_frames)

    canvas = np.zeros_like(sd.skeleton)

    for i in range(n_frames):
        # How many pixels revealed so far
        revealed = min(total_pixels, (i + 1) * pixels_per_frame)

        # Reveal new pixels
        for j in range(max(0, revealed - pixels_per_frame), revealed):
            if j < total_pixels:
                y, x = trace[j]
                canvas[y, x] = 255

        # Dilate slightly for visibility (the skeleton is 1px wide)
        display = cv2.dilate(canvas, np.ones((2, 2), np.uint8), iterations=1)

        frame = make_frame(sd)
        place_image(frame, display, sd.cx, sd.cy)

        # Label
        pct = revealed / total_pixels * 100
        put_text_gray(frame, f"Skeleton: {revealed}/{total_pixels} pixels ({pct:.0f}%)",
                      (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_02_connected_components(sd):
    """Connected Components — jigsaw separation."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Connected Components", f"{sd.n_components} independent pieces")

    centroids = sd.comp_centroids[1:]  # skip background
    n_comp = len(sd.comp_masks)

    for i in range(n_frames):
        t = i / n_frames  # 0 to 1

        frame = make_frame(sd)

        # Animation phases:
        # 0-0.3: drift apart
        # 0.3-0.5: orbit
        # 0.5-0.7: drift back
        # 0.7-1.0: hold together

        for c_idx, mask in enumerate(sd.comp_masks):
            if t < 0.3:
                # Drift apart
                drift = ease_in_out(t / 0.3)
                angle = c_idx * 2 * np.pi / n_comp
                dx = int(drift * 150 * np.cos(angle))
                dy = int(drift * 100 * np.sin(angle))
            elif t < 0.5:
                # Orbit
                orbit_t = (t - 0.3) / 0.2
                angle = c_idx * 2 * np.pi / n_comp + orbit_t * np.pi
                dx = int(150 * np.cos(angle))
                dy = int(100 * np.sin(angle))
            elif t < 0.7:
                # Drift back
                drift = 1.0 - ease_in_out((t - 0.5) / 0.2)
                angle = c_idx * 2 * np.pi / n_comp + np.pi
                dx = int(drift * 150 * np.cos(angle))
                dy = int(drift * 100 * np.sin(angle))
            else:
                dx, dy = 0, 0

            place_image(frame, mask, sd.cx + dx, sd.cy + dy)

        put_text_gray(frame, f"Components: {n_comp}",
                      (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_03_skeleton_vs_flesh(sd):
    """Skeleton vs Flesh — skeleton overlay toggle."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Skeleton vs Flesh", "What skeletonization discards")

    skel_display = cv2.dilate(sd.skeleton, np.ones((2, 2), np.uint8), iterations=1)

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        # Cycle: full(0-0.2) -> skeleton(0.2-0.4) -> flesh(0.4-0.6) -> both(0.6-0.8) -> full(0.8-1.0)
        cycle_t = (t * 3) % 1.0  # repeat 3 times

        if cycle_t < 0.2:
            # Full sigil
            alpha = 1.0
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, alpha)
            label = "Full Sigil"
        elif cycle_t < 0.4:
            # Crossfade to skeleton
            blend = ease_in_out((cycle_t - 0.2) / 0.1) if cycle_t < 0.3 else 1.0
            img = cv2.addWeighted(sd.binary, 1 - blend, skel_display, blend, 0)
            place_image(frame, img, sd.cx, sd.cy)
            label = "Skeleton" if blend > 0.5 else "Full -> Skeleton"
        elif cycle_t < 0.6:
            # Crossfade to flesh
            blend = ease_in_out((cycle_t - 0.4) / 0.1) if cycle_t < 0.5 else 1.0
            img = cv2.addWeighted(skel_display, 1 - blend, sd.flesh, blend, 0)
            place_image(frame, img, sd.cx, sd.cy)
            label = "Flesh" if blend > 0.5 else "Skeleton -> Flesh"
        elif cycle_t < 0.8:
            # Both overlaid
            blend = ease_in_out((cycle_t - 0.6) / 0.1)
            place_image_alpha(frame, sd.flesh, sd.cx, sd.cy, 0.6)
            place_image_alpha(frame, skel_display, sd.cx, sd.cy, 0.8)
            label = "Skeleton + Flesh"
        else:
            # Back to full
            place_image(frame, sd.binary, sd.cx, sd.cy)
            label = "Full Sigil"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_04_junctions_endpoints(sd):
    """Junctions and Endpoints — node highlight."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Junctions & Endpoints", f"{len(sd.junction_clusters)} junctions, {len(sd.endpoint_positions)} endpoints")

    junctions = sd.junction_clusters
    endpoints = sd.endpoint_positions
    n_j = len(junctions)
    n_e = len(endpoints)

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        if t < 0.4:
            # Show sigil, light up junctions one at a time
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, 0.3)
            n_show = int(t / 0.4 * n_j) + 1
            for j_idx in range(min(n_show, n_j)):
                jy, jx = junctions[j_idx]
                fy = sd.cy - sd.display_h // 2 + jy
                fx = sd.cx - sd.display_w // 2 + jx
                cv2.circle(frame, (fx, fy), 5, 255, -1)
            label = f"Junctions: {min(n_show, n_j)}/{n_j}"

        elif t < 0.6:
            # Show all junctions, light up endpoints
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, 0.2)
            for jy, jx in junctions:
                fy = sd.cy - sd.display_h // 2 + jy
                fx = sd.cx - sd.display_w // 2 + jx
                cv2.circle(frame, (fx, fy), 4, 180, -1)

            n_show = int((t - 0.4) / 0.2 * n_e) + 1
            for e_idx in range(min(n_show, n_e)):
                ey, ex = endpoints[e_idx]
                fy = sd.cy - sd.display_h // 2 + ey
                fx = sd.cx - sd.display_w // 2 + ex
                cv2.circle(frame, (fx, fy), 6, 255, 2)
            label = f"Endpoints: {min(n_show, n_e)}/{n_e}"

        elif t < 0.8:
            # Show graph: connect junctions with lines
            blend = ease_in_out((t - 0.6) / 0.1) if t < 0.7 else 1.0
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, 0.15 * (1 - blend))

            # Draw branches as lines
            branch_alpha = min(1.0, (t - 0.6) / 0.1)
            for branch in sd.branches:
                if len(branch) >= 2:
                    start = branch[0]
                    end = branch[-1]
                    p1 = (sd.cx - sd.display_w // 2 + start[1],
                          sd.cy - sd.display_h // 2 + start[0])
                    p2 = (sd.cx - sd.display_w // 2 + end[1],
                          sd.cy - sd.display_h // 2 + end[0])
                    cv2.line(frame, p1, p2, int(200 * branch_alpha), 1)

            # Draw nodes
            for jy, jx in junctions:
                cv2.circle(frame, (sd.cx - sd.display_w // 2 + jx,
                                   sd.cy - sd.display_h // 2 + jy), 4, 255, -1)
            for ey, ex in endpoints:
                cv2.circle(frame, (sd.cx - sd.display_w // 2 + ex,
                                   sd.cy - sd.display_h // 2 + ey), 4, 200, 2)
            label = "Graph representation"

        else:
            # Dissolve back to sigil
            blend = ease_in_out((t - 0.8) / 0.2)
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, blend)
            for jy, jx in junctions:
                cv2.circle(frame, (sd.cx - sd.display_w // 2 + jx,
                                   sd.cy - sd.display_h // 2 + jy),
                           4, int(255 * (1 - blend)), -1)
            label = "Dissolving back"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_05_branch_by_branch(sd):
    """Branch-by-Branch — skeleton decomposition."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Branch by Branch", f"{len(sd.branches)} skeleton branches")

    # Sort branches by length
    sorted_branches = sorted(sd.branches, key=len)
    n_branches = len(sorted_branches)

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        if t < 0.5:
            # Reveal branches one at a time (sorted by length)
            n_show = int(t / 0.5 * n_branches) + 1
            for b_idx in range(min(n_show, n_branches)):
                branch = sorted_branches[b_idx]
                for py, px in branch:
                    fy = sd.cy - sd.display_h // 2 + py
                    fx = sd.cx - sd.display_w // 2 + px
                    if 0 <= fy < HEIGHT and 0 <= fx < WIDTH:
                        frame[fy, fx] = 255
            # Dilate for visibility
            roi_y1 = sd.cy - sd.display_h // 2
            roi_x1 = sd.cx - sd.display_w // 2
            roi = frame[max(0,roi_y1):roi_y1+sd.display_h,
                        max(0,roi_x1):roi_x1+sd.display_w]
            dilated = cv2.dilate(roi, np.ones((2,2), np.uint8))
            frame[max(0,roi_y1):roi_y1+sd.display_h,
                  max(0,roi_x1):roi_x1+sd.display_w] = dilated
            label = f"Branches: {min(n_show, n_branches)}/{n_branches} (by length)"

        elif t < 0.7:
            # All branches visible, scatter them
            scatter = ease_in_out((t - 0.5) / 0.2)
            np.random.seed(42)
            for b_idx, branch in enumerate(sorted_branches):
                angle = np.random.uniform(0, 2 * np.pi)
                dist = scatter * 80
                dx = int(dist * np.cos(angle))
                dy = int(dist * np.sin(angle))
                for py, px in branch:
                    fy = sd.cy - sd.display_h // 2 + py + dy
                    fx = sd.cx - sd.display_w // 2 + px + dx
                    if 0 <= fy < HEIGHT and 0 <= fx < WIDTH:
                        frame[fy, fx] = 255
            label = "Branches scattered"

        else:
            # Reassemble
            scatter = 1.0 - ease_in_out((t - 0.7) / 0.3)
            np.random.seed(42)
            for b_idx, branch in enumerate(sorted_branches):
                angle = np.random.uniform(0, 2 * np.pi)
                dist = scatter * 80
                dx = int(dist * np.cos(angle))
                dy = int(dist * np.sin(angle))
                for py, px in branch:
                    fy = sd.cy - sd.display_h // 2 + py + dy
                    fx = sd.cx - sd.display_w // 2 + px + dx
                    if 0 <= fy < HEIGHT and 0 <= fx < WIDTH:
                        frame[fy, fx] = 255
            label = "Reassembling"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_06_holes(sd):
    """Holes — filling and emptying."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Holes", f"{len(sd.hole_contours)} enclosed regions")

    n_holes = len(sd.hole_contours)

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        if t < 0.4:
            # Show sigil, fill holes one at a time
            place_image(frame, sd.binary, sd.cx, sd.cy)
            n_fill = int(t / 0.4 * n_holes) + 1
            hole_overlay = np.zeros_like(sd.binary)
            for h_idx in range(min(n_fill, n_holes)):
                cv2.drawContours(hole_overlay, [sd.hole_contours[h_idx]], -1, 200, -1)
            place_image(frame, hole_overlay, sd.cx, sd.cy)
            label = f"Filling holes: {min(n_fill, n_holes)}/{n_holes}"

        elif t < 0.6:
            # All holes filled
            place_image(frame, sd.binary, sd.cx, sd.cy)
            hole_overlay = np.zeros_like(sd.binary)
            for h in sd.hole_contours:
                cv2.drawContours(hole_overlay, [h], -1, 200, -1)
            place_image(frame, hole_overlay, sd.cx, sd.cy)
            label = f"All {n_holes} holes filled"

        elif t < 0.8:
            # INVERSE: only holes visible
            blend = ease_in_out((t - 0.6) / 0.1) if t < 0.7 else 1.0
            sigil_alpha = 1.0 - blend
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, sigil_alpha * 0.3)
            hole_overlay = np.zeros_like(sd.binary)
            for h in sd.hole_contours:
                cv2.drawContours(hole_overlay, [h], -1, 255, -1)
            place_image(frame, hole_overlay, sd.cx, sd.cy)
            label = "Negative space only"

        else:
            # Fade back to sigil
            blend = ease_in_out((t - 0.8) / 0.2)
            hole_overlay = np.zeros_like(sd.binary)
            for h in sd.hole_contours:
                cv2.drawContours(hole_overlay, [h], -1, int(255 * (1 - blend)), -1)
            place_image(frame, hole_overlay, sd.cx, sd.cy)
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, blend)
            label = "Returning to whole"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_07_hough_lines(sd):
    """Hough Lines — detected geometry overlay."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Hough Lines", f"{len(sd.hough_lines)} detected line segments")

    n_lines = len(sd.hough_lines)
    ox = sd.cx - sd.display_w // 2
    oy = sd.cy - sd.display_h // 2

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        if t < 0.4:
            # Sigil with lines fading in one at a time
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, 0.4)
            n_show = int(t / 0.4 * n_lines) + 1
            for l_idx in range(min(n_show, n_lines)):
                x1, y1, x2, y2 = sd.hough_lines[l_idx][0]
                cv2.line(frame, (ox + x1, oy + y1), (ox + x2, oy + y2), 220, 1)
            label = f"Lines: {min(n_show, n_lines)}/{n_lines}"

        elif t < 0.6:
            # Lines only, sigil fades out
            fade = 1.0 - ease_in_out((t - 0.4) / 0.1) if t < 0.5 else 0.0
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, fade * 0.4)
            for line in sd.hough_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (ox + x1, oy + y1), (ox + x2, oy + y2), 220, 1)
            label = "Geometric skeleton"

        elif t < 0.8:
            # Lines splay apart by angle
            splay = ease_in_out((t - 0.6) / 0.2)
            cx_local = sd.display_w / 2
            cy_local = sd.display_h / 2
            for line in sd.hough_lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                push = splay * 40
                dx = int(push * np.cos(angle + np.pi / 2))
                dy = int(push * np.sin(angle + np.pi / 2))
                cv2.line(frame, (ox + x1 + dx, oy + y1 + dy),
                         (ox + x2 + dx, oy + y2 + dy), 220, 1)
            label = "Lines splayed by angle"

        else:
            # Snap back + sigil returns
            blend = ease_in_out((t - 0.8) / 0.2)
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, blend * 0.4)
            splay = 1.0 - ease_in_out((t - 0.8) / 0.15)
            for line in sd.hough_lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                push = splay * 40
                dx = int(push * np.cos(angle + np.pi / 2))
                dy = int(push * np.sin(angle + np.pi / 2))
                cv2.line(frame, (ox + x1 + dx, oy + y1 + dy),
                         (ox + x2 + dx, oy + y2 + dy), int(220 * (1 - blend * 0.5)), 1)
            label = "Returning"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_08_hough_circles(sd):
    """Hough Circles — circular detection."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Hough Circles", f"{len(sd.hough_circles)} detected circles")

    n_circ = len(sd.hough_circles)
    ox = sd.cx - sd.display_w // 2
    oy = sd.cy - sd.display_h // 2

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        if t < 0.35:
            # Sigil with circles appearing
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, 0.4)
            n_show = int(t / 0.35 * n_circ) + 1
            for c_idx in range(min(n_show, n_circ)):
                cx_c, cy_c, r = sd.hough_circles[c_idx]
                cv2.circle(frame, (ox + cx_c, oy + cy_c), r, 220, 1)
            label = f"Circles: {min(n_show, n_circ)}/{n_circ}"

        elif t < 0.55:
            # Circles only, pulsing
            pulse = 1.0 + 0.2 * np.sin(2 * np.pi * (t - 0.35) / 0.1)
            for c_idx in range(n_circ):
                cx_c, cy_c, r = sd.hough_circles[c_idx]
                cv2.circle(frame, (ox + cx_c, oy + cy_c),
                           int(r * pulse), 220, 1)
            label = "Circles pulsing"

        elif t < 0.75:
            # Circles drift to cluster
            drift = ease_in_out((t - 0.55) / 0.2)
            target_x = sd.display_w // 2
            target_y = sd.display_h // 2
            for c_idx in range(n_circ):
                cx_c, cy_c, r = sd.hough_circles[c_idx]
                new_cx = int(lerp(cx_c, target_x, drift * 0.7))
                new_cy = int(lerp(cy_c, target_y, drift * 0.7))
                cv2.circle(frame, (ox + new_cx, oy + new_cy), r, 220, 1)
            label = "Circles clustering"

        else:
            # Return + sigil
            drift = 1.0 - ease_in_out((t - 0.75) / 0.25)
            blend = ease_in_out((t - 0.75) / 0.25)
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, blend * 0.5)
            target_x = sd.display_w // 2
            target_y = sd.display_h // 2
            for c_idx in range(n_circ):
                cx_c, cy_c, r = sd.hough_circles[c_idx]
                new_cx = int(lerp(cx_c, target_x, drift * 0.7))
                new_cy = int(lerp(cy_c, target_y, drift * 0.7))
                cv2.circle(frame, (ox + new_cx, oy + new_cy), r,
                           int(220 * (1 - blend * 0.5)), 1)
            label = "Returning"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_09_angle_histogram(sd):
    """Angle Histogram — directional decomposition."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Angle Histogram", "Directional decomposition (12 bins x 15 degrees)")

    bin_labels = [f"{b*15}-{(b+1)*15}" for b in range(12)]

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        if t < 0.5:
            # Show each angle bin one at a time
            bin_idx = int(t / 0.5 * 12) % 12
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, 0.15)
            place_image(frame, sd.angle_masks[bin_idx], sd.cx, sd.cy)
            label = f"Angle bin: {bin_labels[bin_idx]} degrees"

        elif t < 0.7:
            # Rapid cycle through all bins
            cycle_speed = 20  # bins per second
            bin_idx = int((t - 0.5) * cycle_speed * FPS / n_frames * 12) % 12
            place_image(frame, sd.angle_masks[bin_idx], sd.cx, sd.cy)
            label = f"Cycling: {bin_labels[bin_idx]}"

        else:
            # Only dominant bins (0° and 90°) remain
            fade = ease_in_out((t - 0.7) / 0.1) if t < 0.8 else 1.0
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, 0.1)
            # bin 0 = 0-15°, bin 6 = 90-105°
            place_image_alpha(frame, sd.angle_masks[0], sd.cx, sd.cy, fade)
            place_image_alpha(frame, sd.angle_masks[6], sd.cx, sd.cy, fade)
            label = "Dominant: 0 and 90 degrees"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_10_radial_profile(sd):
    """Radial Profile — ring-by-ring reveal."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Radial Profile", "8 concentric rings, center to edge")

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        if t < 0.4:
            # Reveal rings one at a time from center
            n_show = int(t / 0.4 * 8) + 1
            for r_idx in range(min(n_show, 8)):
                place_image(frame, sd.radial_masks[r_idx], sd.cx, sd.cy)
            label = f"Ring {min(n_show, 8)}/8 (center to edge)"

        elif t < 0.7:
            # Radar sweep
            sweep_speed = 2.0  # sweeps per segment
            sweep_pos = ((t - 0.4) / 0.3 * sweep_speed) % 1.0
            active_ring = int(sweep_pos * 8) % 8
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, 0.15)
            place_image(frame, sd.radial_masks[active_ring], sd.cx, sd.cy)
            # Fading trail
            for trail in range(1, 3):
                prev_ring = (active_ring - trail) % 8
                place_image_alpha(frame, sd.radial_masks[prev_ring], sd.cx, sd.cy,
                                  0.5 / (trail + 1))
            label = f"Radar sweep: ring {active_ring + 1}"

        else:
            # Highlight densest rings
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, 0.15)
            # Rings 1 and 3 have highest density per the radial profile
            place_image(frame, sd.radial_masks[1], sd.cx, sd.cy)
            place_image(frame, sd.radial_masks[3], sd.cx, sd.cy)
            label = "Densest rings: 2 and 4 (density 0.405, 0.379)"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_11_quadrant_density(sd):
    """Quadrant Density — four-piece jigsaw."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Quadrant Density", "Four-piece jigsaw (TL=0.313, TR=0.328, BL=0.303, BR=0.282)")

    q_labels = ["TL", "TR", "BL", "BR"]
    q_offsets_base = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        if t < 0.3:
            # Separate quadrants
            sep = ease_in_out(t / 0.3) * 60
            for q_idx, mask in enumerate(sd.quadrant_masks):
                dy_sign, dx_sign = q_offsets_base[q_idx]
                dx = int(dx_sign * sep)
                dy = int(dy_sign * sep)
                place_image(frame, mask, sd.cx + dx, sd.cy + dy)
            label = "Quadrants separating"

        elif t < 0.6:
            # Rotate individual quadrants
            rot_t = (t - 0.3) / 0.3
            rot_angles = [int(rot_t * 360) % 360, int(rot_t * 270) % 360,
                         int(rot_t * 180) % 360, int(rot_t * 90) % 360]
            for q_idx, mask in enumerate(sd.quadrant_masks):
                dy_sign, dx_sign = q_offsets_base[q_idx]
                dx = int(dx_sign * 60)
                dy = int(dy_sign * 60)
                # Rotate the quadrant
                h, w = mask.shape
                M = cv2.getRotationMatrix2D((w/2, h/2), rot_angles[q_idx], 1.0)
                rotated = cv2.warpAffine(mask, M, (w, h),
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                place_image(frame, rotated, sd.cx + dx, sd.cy + dy)
            label = "Quadrants rotating"

        elif t < 0.8:
            # Flip quadrants
            flip_t = (t - 0.6) / 0.2
            for q_idx, mask in enumerate(sd.quadrant_masks):
                dy_sign, dx_sign = q_offsets_base[q_idx]
                dx = int(dx_sign * 60)
                dy = int(dy_sign * 60)
                # Cycle through flips
                flip_phase = int(flip_t * 4) % 4
                if flip_phase == 0:
                    show = mask
                elif flip_phase == 1:
                    show = cv2.flip(mask, 1)  # horizontal
                elif flip_phase == 2:
                    show = cv2.flip(mask, 0)  # vertical
                else:
                    show = cv2.flip(mask, -1)  # both
                place_image(frame, show, sd.cx + dx, sd.cy + dy)
            label = "Quadrants flipping"

        else:
            # Reassemble
            sep = (1.0 - ease_in_out((t - 0.8) / 0.2)) * 60
            for q_idx, mask in enumerate(sd.quadrant_masks):
                dy_sign, dx_sign = q_offsets_base[q_idx]
                dx = int(dx_sign * sep)
                dy = int(dy_sign * sep)
                place_image(frame, mask, sd.cx + dx, sd.cy + dy)
            label = "Reassembling"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_12_h_symmetry(sd):
    """Horizontal Symmetry — mirror test."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Horizontal Symmetry", "Score: 0.642 (left-right mirror)")

    h_flip = cv2.flip(sd.binary, 1)
    # Overlap = where both original and flip have ink
    overlap = cv2.bitwise_and(sd.binary, h_flip)
    # Difference
    diff = cv2.absdiff(sd.binary, h_flip)

    mid_x = sd.display_w // 2

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        if t < 0.25:
            # Show original
            place_image(frame, sd.binary, sd.cx, sd.cy)
            label = "Original"

        elif t < 0.45:
            # Overlay mirror at increasing opacity
            alpha = ease_in_out((t - 0.25) / 0.1) if t < 0.35 else 1.0
            place_image(frame, sd.binary, sd.cx, sd.cy)
            place_image_alpha(frame, h_flip, sd.cx, sd.cy, alpha * 0.5)
            label = "Overlaid with horizontal mirror"

        elif t < 0.6:
            # Highlight overlap
            place_image_alpha(frame, diff, sd.cx, sd.cy, 0.3)
            place_image(frame, overlap, sd.cx, sd.cy)
            label = f"Overlap regions (symmetry = 0.642)"

        elif t < 0.8:
            # Left half original, right half mirrored
            composite = sd.binary.copy()
            left_mirror = cv2.flip(sd.binary[:, :mid_x], 1)
            # Fit right side with mirror of left
            rw = min(left_mirror.shape[1], sd.display_w - mid_x)
            blend = ease_in_out((t - 0.6) / 0.1) if t < 0.7 else 1.0
            right_region = composite[:, mid_x:mid_x + rw]
            composite[:, mid_x:mid_x + rw] = cv2.addWeighted(
                right_region, 1 - blend, left_mirror[:, :rw], blend, 0)
            place_image(frame, composite, sd.cx, sd.cy)
            label = "Right = mirror of left"

        else:
            # Return to original
            blend = ease_in_out((t - 0.8) / 0.2)
            composite = sd.binary.copy()
            left_mirror = cv2.flip(sd.binary[:, :mid_x], 1)
            rw = min(left_mirror.shape[1], sd.display_w - mid_x)
            composite[:, mid_x:mid_x + rw] = cv2.addWeighted(
                left_mirror[:, :rw], 1 - blend,
                sd.binary[:, mid_x:mid_x + rw], blend, 0)
            place_image(frame, composite, sd.cx, sd.cy)
            label = "Returning to original"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_13_v_symmetry(sd):
    """Vertical Symmetry — top-bottom mirror."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Vertical Symmetry", "Score: 0.141 (top-bottom mirror — very low)")

    v_flip = cv2.flip(sd.binary, 0)
    overlap = cv2.bitwise_and(sd.binary, v_flip)
    diff = cv2.absdiff(sd.binary, v_flip)
    mid_y = sd.display_h // 2

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        if t < 0.25:
            place_image(frame, sd.binary, sd.cx, sd.cy)
            label = "Original"

        elif t < 0.45:
            alpha = ease_in_out((t - 0.25) / 0.1) if t < 0.35 else 1.0
            place_image(frame, sd.binary, sd.cx, sd.cy)
            place_image_alpha(frame, v_flip, sd.cx, sd.cy, alpha * 0.5)
            label = "Overlaid with vertical mirror"

        elif t < 0.6:
            place_image_alpha(frame, diff, sd.cx, sd.cy, 0.3)
            place_image(frame, overlap, sd.cx, sd.cy)
            label = f"Overlap regions (symmetry = 0.141) — mostly mismatch!"

        elif t < 0.8:
            composite = sd.binary.copy()
            top_mirror = cv2.flip(sd.binary[:mid_y, :], 0)
            bh = min(top_mirror.shape[0], sd.display_h - mid_y)
            blend = ease_in_out((t - 0.6) / 0.1) if t < 0.7 else 1.0
            composite[mid_y:mid_y + bh, :] = cv2.addWeighted(
                composite[mid_y:mid_y + bh, :], 1 - blend,
                top_mirror[:bh, :], blend, 0)
            place_image(frame, composite, sd.cx, sd.cy)
            label = "Bottom = mirror of top (forced symmetry)"

        else:
            blend = ease_in_out((t - 0.8) / 0.2)
            composite = sd.binary.copy()
            top_mirror = cv2.flip(sd.binary[:mid_y, :], 0)
            bh = min(top_mirror.shape[0], sd.display_h - mid_y)
            composite[mid_y:mid_y + bh, :] = cv2.addWeighted(
                top_mirror[:bh, :], 1 - blend,
                sd.binary[mid_y:mid_y + bh, :], blend, 0)
            place_image(frame, composite, sd.cx, sd.cy)
            label = "Returning to original"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_14_fractal_dimension(sd):
    """Fractal Dimension — box-counting visualization."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Fractal Dimension", "Box-counting: FD = 1.546")

    box_sizes = [64, 32, 16, 8, 4]
    ox = sd.cx - sd.display_w // 2
    oy = sd.cy - sd.display_h // 2

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        # Show sigil faintly
        place_image_alpha(frame, sd.binary, sd.cx, sd.cy, 0.25)

        # Which box size are we showing
        size_idx = min(int(t * len(box_sizes) * 1.2), len(box_sizes) - 1)
        box_size = box_sizes[size_idx]

        # Count and draw boxes that contain ink
        count = 0
        for by in range(0, sd.display_h, box_size):
            for bx in range(0, sd.display_w, box_size):
                # Check if this box contains any ink
                box_region = sd.binary[by:by+box_size, bx:bx+box_size]
                if np.any(box_region > 0):
                    count += 1
                    # Draw box outline on frame
                    cv2.rectangle(frame,
                                  (ox + bx, oy + by),
                                  (ox + bx + box_size, oy + by + box_size),
                                  180, 1)

        label = f"Box size: {box_size}px -> {count} boxes contain ink (FD = 1.546)"
        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_15_erosion_dilation(sd):
    """Ink Ratio & Compactness — erosion/dilation."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Ink Ratio & Compactness", "Ink = 30.6%, Compactness = 0.338")

    max_erosion = 8
    max_dilation = 12

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        if t < 0.45:
            # Progressive erosion
            erosion_level = int(t / 0.45 * max_erosion)
            if erosion_level > 0:
                kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(sd.binary, kernel, iterations=erosion_level)
            else:
                eroded = sd.binary.copy()
            ink_pct = np.sum(eroded > 0) / eroded.size * 100
            place_image(frame, eroded, sd.cx, sd.cy)
            label = f"Erosion: {erosion_level} iterations, ink = {ink_pct:.1f}%"

        elif t < 0.55:
            # Fully eroded, pause
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(sd.binary, kernel, iterations=max_erosion)
            place_image(frame, eroded, sd.cx, sd.cy)
            label = "Maximum erosion"

        else:
            # Progressive dilation from original
            dilation_level = int((t - 0.55) / 0.45 * max_dilation)
            if dilation_level > 0:
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(sd.binary, kernel, iterations=dilation_level)
            else:
                dilated = sd.binary.copy()
            ink_pct = np.sum(dilated > 0) / dilated.size * 100
            place_image(frame, dilated, sd.cx, sd.cy)
            label = f"Dilation: {dilation_level} iterations, ink = {ink_pct:.1f}%"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_16_contour_nesting(sd):
    """Contour Nesting — peeling layers."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Contour Nesting", f"{len(sd.contours)} contours, peeling layers")

    # Sort contours by area (largest first)
    sorted_contours = sorted(sd.contours, key=cv2.contourArea, reverse=True)
    n_cont = min(len(sorted_contours), 30)  # cap for performance

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)
        ox = sd.cx - sd.display_w // 2
        oy = sd.cy - sd.display_h // 2

        if t < 0.5:
            # Peel off contours one at a time
            n_peel = int(t / 0.5 * n_cont)
            for c_idx in range(n_peel, n_cont):
                c = sorted_contours[c_idx] + np.array([ox, oy])
                cv2.drawContours(frame, [c], -1, 200, 1)
            # The peeled ones float away
            for c_idx in range(n_peel):
                c = sorted_contours[c_idx].copy()
                drift = 30 + c_idx * 5
                angle = c_idx * np.pi / 4
                dx = int(drift * np.cos(angle))
                dy = int(drift * np.sin(angle))
                c_shifted = c + np.array([ox + dx, oy + dy])
                cv2.drawContours(frame, [c_shifted], -1, 120, 1)
            label = f"Peeled: {n_peel}/{n_cont} contours"

        else:
            # Reassemble in reverse order
            n_reassemble = int((t - 0.5) / 0.5 * n_cont)
            for c_idx in range(n_cont):
                c = sorted_contours[c_idx]
                if c_idx < n_reassemble:
                    c_final = c + np.array([ox, oy])
                    cv2.drawContours(frame, [c_final], -1, 200, 1)
                else:
                    drift = max(0, 30 + c_idx * 5 - (t - 0.5) / 0.5 * 50)
                    angle = c_idx * np.pi / 4
                    dx = int(drift * np.cos(angle))
                    dy = int(drift * np.sin(angle))
                    c_shifted = c + np.array([ox + dx, oy + dy])
                    cv2.drawContours(frame, [c_shifted], -1, 120, 1)
            label = f"Reassembling: {n_reassemble}/{n_cont}"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_17_fourier_reconstruction(sd):
    """Fourier Reconstruction — harmonic build-up."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Fourier Reconstruction", "Building shape from harmonics")

    coeffs = sd.fourier_coeffs
    ox = sd.cx - sd.display_w // 2
    oy = sd.cy - sd.display_h // 2

    if coeffs is None or len(coeffs) < 4:
        # Fallback if no Fourier data
        for i in range(n_frames):
            frame = make_frame(sd)
            place_image(frame, sd.binary, sd.cx, sd.cy)
            put_text_gray(frame, "No Fourier data available", (20, HEIGHT-30), 0.5, 100, 1)
            yield frame
        return

    n = len(coeffs)
    harmonic_levels = [1, 2, 4, 8, 16, 32, min(64, n//2)]

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        if t < 0.7:
            # Progressive harmonic build-up
            level_idx = int(t / 0.7 * len(harmonic_levels))
            level_idx = min(level_idx, len(harmonic_levels) - 1)
            n_harmonics = harmonic_levels[level_idx]

            # Reconstruct contour with limited harmonics
            recon_coeffs = np.zeros_like(coeffs)
            recon_coeffs[:n_harmonics] = coeffs[:n_harmonics]
            recon_coeffs[-n_harmonics:] = coeffs[-n_harmonics:]

            z_recon = np.fft.ifft(recon_coeffs)
            pts = np.column_stack([z_recon.real, z_recon.imag]).astype(np.int32)
            pts = pts + np.array([ox, oy])
            pts = pts.reshape(-1, 1, 2)

            cv2.polylines(frame, [pts], True, 220, 1)
            label = f"Harmonics: {n_harmonics} of {n//2}"

        else:
            # Full reconstruction vs original
            blend = ease_in_out((t - 0.7) / 0.3)
            z_recon = np.fft.ifft(coeffs)
            pts = np.column_stack([z_recon.real, z_recon.imag]).astype(np.int32)
            pts = pts + np.array([ox, oy])
            pts = pts.reshape(-1, 1, 2)

            cv2.polylines(frame, [pts], True, int(220 * (1 - blend)), 1)
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, blend)
            label = "Full reconstruction -> original"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_18_graph_topology(sd):
    """Graph Topology — network animation."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Graph Topology", "36 nodes, 36 edges — force-directed layout")

    junctions = sd.junction_clusters
    endpoints = sd.endpoint_positions
    nodes = [(y, x) for y, x in junctions] + [(y, x) for y, x in endpoints]
    n_nodes = len(nodes)
    ox = sd.cx - sd.display_w // 2
    oy = sd.cy - sd.display_h // 2

    # Original positions (in display space)
    orig_pos = np.array(nodes, dtype=np.float32)

    # Compute a simple force-directed layout target
    # Start from original, apply repulsion + spring forces
    target_pos = orig_pos.copy()
    center = np.mean(target_pos, axis=0)
    # Spread out from center
    for _ in range(50):
        for j in range(n_nodes):
            force = np.zeros(2)
            for k in range(n_nodes):
                if j == k:
                    continue
                diff = target_pos[j] - target_pos[k]
                dist = max(np.linalg.norm(diff), 1)
                force += diff / dist * 100 / (dist + 1)
            # Spring back to center
            force -= (target_pos[j] - center) * 0.05
            target_pos[j] += force * 0.1

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        if t < 0.15:
            # Show sigil fading, graph appearing
            blend = ease_in_out(t / 0.15)
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, 1 - blend)

        if t < 0.4:
            # Spatial layout
            blend = min(1.0, t / 0.15)
            for j in range(n_nodes):
                y, x = orig_pos[j]
                cv2.circle(frame, (int(ox + x), int(oy + y)), 3, int(255 * blend), -1)
            for branch in sd.branches:
                if len(branch) >= 2:
                    p1 = (int(ox + branch[0][1]), int(oy + branch[0][0]))
                    p2 = (int(ox + branch[-1][1]), int(oy + branch[-1][0]))
                    cv2.line(frame, p1, p2, int(150 * blend), 1)
            label = "Spatial layout"

        elif t < 0.7:
            # Transition to force-directed
            morph = ease_in_out((t - 0.4) / 0.3)
            current_pos = orig_pos * (1 - morph) + target_pos * morph

            for j in range(n_nodes):
                y, x = current_pos[j]
                cv2.circle(frame, (int(ox + x), int(oy + y)), 3, 255, -1)
            for branch in sd.branches:
                if len(branch) >= 2:
                    s_idx = min(range(n_nodes),
                               key=lambda k: abs(nodes[k][0]-branch[0][0]) + abs(nodes[k][1]-branch[0][1]))
                    e_idx = min(range(n_nodes),
                               key=lambda k: abs(nodes[k][0]-branch[-1][0]) + abs(nodes[k][1]-branch[-1][1]))
                    p1 = (int(ox + current_pos[s_idx][1]), int(oy + current_pos[s_idx][0]))
                    p2 = (int(ox + current_pos[e_idx][1]), int(oy + current_pos[e_idx][0]))
                    cv2.line(frame, p1, p2, 150, 1)
            label = "Morphing to force-directed layout"

        else:
            # Snap back
            morph = 1.0 - ease_in_out((t - 0.7) / 0.2)
            if t > 0.9:
                morph = 0
            current_pos = orig_pos * (1 - morph) + target_pos * morph

            for j in range(n_nodes):
                y, x = current_pos[j]
                cv2.circle(frame, (int(ox + x), int(oy + y)), 3, 255, -1)
            for branch in sd.branches:
                if len(branch) >= 2:
                    s_idx = min(range(n_nodes),
                               key=lambda k: abs(nodes[k][0]-branch[0][0]) + abs(nodes[k][1]-branch[0][1]))
                    e_idx = min(range(n_nodes),
                               key=lambda k: abs(nodes[k][0]-branch[-1][0]) + abs(nodes[k][1]-branch[-1][1]))
                    p1 = (int(ox + current_pos[s_idx][1]), int(oy + current_pos[s_idx][0]))
                    p2 = (int(ox + current_pos[e_idx][1]), int(oy + current_pos[e_idx][0]))
                    cv2.line(frame, p1, p2, 150, 1)

            if t > 0.9:
                place_image_alpha(frame, sd.binary, sd.cx, sd.cy,
                                  ease_in_out((t - 0.9) / 0.1))
            label = "Returning to spatial" if t < 0.9 else "Back to sigil"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_19_terminal_decorations(sd):
    """Terminal Decorations — endpoint catalog."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Terminal Decorations", "18 endpoints: 12 filled/cross, 1 circle, 5 simple")

    endpoints = sd.endpoint_positions
    n_ep = len(endpoints)
    frames_per_endpoint = n_frames * 3 // (4 * max(n_ep, 1))  # 75% of time for catalog
    ox = sd.cx - sd.display_w // 2
    oy = sd.cy - sd.display_h // 2
    zoom_size = 30  # pixels around endpoint to zoom

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        if t < 0.75 and n_ep > 0:
            # Cycle through endpoints
            ep_idx = min(int(t / 0.75 * n_ep), n_ep - 1)
            ey, ex = endpoints[ep_idx]

            # Show full sigil with highlight circle
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, 0.4)
            cv2.circle(frame, (ox + ex, oy + ey), 10, 255, 2)

            # Show zoomed view in corner
            y1 = max(0, ey - zoom_size)
            y2 = min(sd.display_h, ey + zoom_size)
            x1 = max(0, ex - zoom_size)
            x2 = min(sd.display_w, ex + zoom_size)

            if y2 > y1 and x2 > x1:
                zoomed = sd.binary[y1:y2, x1:x2]
                zoomed_big = cv2.resize(zoomed, (200, 200), interpolation=cv2.INTER_NEAREST)
                # Place in top-right corner
                frame[50:250, WIDTH - 250:WIDTH - 50] = np.maximum(
                    frame[50:250, WIDTH - 250:WIDTH - 50], zoomed_big)
                cv2.rectangle(frame, (WIDTH - 252, 48), (WIDTH - 48, 252), 150, 1)

            label = f"Endpoint {ep_idx + 1}/{n_ep} at ({ex}, {ey})"

        else:
            # All endpoints blink by type
            blink_phase = int((t - 0.75) / 0.05) % 3 if t >= 0.75 else 0
            place_image_alpha(frame, sd.binary, sd.cx, sd.cy, 0.3)

            for ep_idx, (ey, ex) in enumerate(endpoints):
                # Simple classification based on local ink density
                y1 = max(0, ey - 5)
                y2 = min(sd.display_h, ey + 5)
                x1 = max(0, ex - 5)
                x2 = min(sd.display_w, ex + 5)
                local_ink = np.sum(sd.binary[y1:y2, x1:x2] > 0)

                if local_ink > 60:
                    ep_type = 0  # filled/cross
                elif local_ink > 30:
                    ep_type = 1  # circle
                else:
                    ep_type = 2  # simple

                if ep_type == blink_phase:
                    cv2.circle(frame, (ox + ex, oy + ey), 6, 255, -1)
                else:
                    cv2.circle(frame, (ox + ex, oy + ey), 4, 120, 1)

            type_names = ["filled/cross", "circle", "simple"]
            label = f"Blinking: {type_names[blink_phase]} terminals"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


def segment_20_recombination(sd):
    """Recombination — grand finale shuffle & reassemble."""
    n_frames = int((SEGMENT_DURATION - TITLE_CARD_DURATION) * FPS)
    yield from title_card("Recombination", "Disassemble and reassemble")

    for i in range(n_frames):
        t = i / n_frames
        frame = make_frame(sd)

        if t < 0.15:
            # Start with whole sigil
            place_image(frame, sd.binary, sd.cx, sd.cy)
            label = "Whole"

        elif t < 0.3:
            # Break into components
            sep = ease_in_out((t - 0.15) / 0.15) * 100
            for c_idx, mask in enumerate(sd.comp_masks):
                angle = c_idx * 2 * np.pi / len(sd.comp_masks)
                dx = int(sep * np.cos(angle))
                dy = int(sep * np.sin(angle))
                place_image(frame, mask, sd.cx + dx, sd.cy + dy)
            label = "Breaking into components"

        elif t < 0.45:
            # Break into quadrants
            sep = ease_in_out((t - 0.3) / 0.15) * 80
            q_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for q_idx, mask in enumerate(sd.quadrant_masks):
                dy_sign, dx_sign = q_offsets[q_idx]
                dx = int(dx_sign * sep)
                dy = int(dy_sign * sep)
                place_image(frame, mask, sd.cx + dx, sd.cy + dy)
            label = "Breaking into quadrants"

        elif t < 0.55:
            # Scatter and rotate
            scatter = ease_in_out((t - 0.45) / 0.1) * 50
            np.random.seed(123)
            for q_idx, mask in enumerate(sd.quadrant_masks):
                angle = np.random.uniform(0, 2 * np.pi)
                rot = int(scatter * 3)
                dx = int(80 * np.cos(q_idx * np.pi/2) + scatter * np.cos(angle))
                dy = int(80 * np.sin(q_idx * np.pi/2) + scatter * np.sin(angle))
                h, w = mask.shape
                M = cv2.getRotationMatrix2D((w/2, h/2), rot * (q_idx + 1), 1.0)
                rotated = cv2.warpAffine(mask, M, (w, h), borderValue=0)
                place_image(frame, rotated, sd.cx + dx, sd.cy + dy)
            label = "Maximum chaos"

        elif t < 0.7:
            # Begin reassembly — quadrants return
            progress = ease_in_out((t - 0.55) / 0.15)
            sep = (1.0 - progress) * 80
            q_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for q_idx, mask in enumerate(sd.quadrant_masks):
                dy_sign, dx_sign = q_offsets[q_idx]
                rot = int((1 - progress) * 50 * (q_idx + 1))
                dx = int(dx_sign * sep)
                dy = int(dy_sign * sep)
                h, w = mask.shape
                M = cv2.getRotationMatrix2D((w/2, h/2), rot, 1.0)
                rotated = cv2.warpAffine(mask, M, (w, h), borderValue=0)
                place_image(frame, rotated, sd.cx + dx, sd.cy + dy)
            label = "Quadrants returning"

        elif t < 0.85:
            # Components merge
            progress = ease_in_out((t - 0.7) / 0.15)
            sep = (1.0 - progress) * 100
            for c_idx, mask in enumerate(sd.comp_masks):
                angle = c_idx * 2 * np.pi / len(sd.comp_masks)
                dx = int(sep * np.cos(angle))
                dy = int(sep * np.sin(angle))
                place_image(frame, mask, sd.cx + dx, sd.cy + dy)
            label = "Components merging"

        else:
            # Hold on complete sigil
            place_image(frame, sd.binary, sd.cx, sd.cy)
            label = "#11 Gusion — Complete"

        put_text_gray(frame, label, (20, HEIGHT - 30), 0.5, 100, 1)
        yield frame

    return


# ============================================================
# MAIN
# ============================================================

SEGMENTS = [
    ("01", "Skeleton Trace", segment_01_skeleton_trace),
    ("02", "Connected Components", segment_02_connected_components),
    ("03", "Skeleton vs Flesh", segment_03_skeleton_vs_flesh),
    ("04", "Junctions & Endpoints", segment_04_junctions_endpoints),
    ("05", "Branch by Branch", segment_05_branch_by_branch),
    ("06", "Holes", segment_06_holes),
    ("07", "Hough Lines", segment_07_hough_lines),
    ("08", "Hough Circles", segment_08_hough_circles),
    ("09", "Angle Histogram", segment_09_angle_histogram),
    ("10", "Radial Profile", segment_10_radial_profile),
    ("11", "Quadrant Density", segment_11_quadrant_density),
    ("12", "H-Symmetry", segment_12_h_symmetry),
    ("13", "V-Symmetry", segment_13_v_symmetry),
    ("14", "Fractal Dimension", segment_14_fractal_dimension),
    ("15", "Erosion / Dilation", segment_15_erosion_dilation),
    ("16", "Contour Nesting", segment_16_contour_nesting),
    ("17", "Fourier Reconstruction", segment_17_fourier_reconstruction),
    ("18", "Graph Topology", segment_18_graph_topology),
    ("19", "Terminal Decorations", segment_19_terminal_decorations),
    ("20", "Recombination", segment_20_recombination),
]


def main():
    print("=" * 60)
    print("GUSION DEEP-DIVE: 20-MINUTE ANALYTICAL ANIMATION")
    print("=" * 60)

    sd = preprocess()

    output_path = str(OUTDIR / "gusion_deep_dive.mp4")

    total_frames = 0
    start_time = time.time()

    print(f"Rendering {len(SEGMENTS)} segments ({SEGMENT_DURATION}s each)...")
    print(f"Streaming frames directly to ffmpeg (pipe-based encoding)\n")

    # Start ffmpeg process with pipe input
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
        for seg_num, seg_name, seg_func in SEGMENTS:
            print(f"  Segment {seg_num}: {seg_name}...", end="", flush=True)
            t0 = time.time()

            seg_frames = seg_func(sd)
            seg_count = 0

            for frame_gray in seg_frames:
                # Convert grayscale to BGR for video encoding
                frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
                proc.stdin.write(frame_bgr.tobytes())
                total_frames += 1
                seg_count += 1

            elapsed = time.time() - t0
            print(f" {seg_count} frames ({elapsed:.1f}s)")

        # Clean shutdown: close stdin so ffmpeg can finalize the file
        proc.stdin.close()
        print(f"\nWaiting for ffmpeg to finalize MP4...")
        stdout, stderr = proc.communicate(timeout=600)

        if proc.returncode != 0:
            print(f"ffmpeg error: {stderr.decode('utf-8', errors='replace')[-500:]}")
        else:
            print(f"ffmpeg finished successfully")

    except Exception as e:
        print(f"\nError during rendering: {e}")
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
