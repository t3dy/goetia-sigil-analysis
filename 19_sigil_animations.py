"""
Script 19: Goetic Sigil Animation Video Generator
Creates a YouTube-ready MP4 video with animated effects for all 72 sigils.

Effects include:
  - Color cycling (psychedelic rainbow wash)
  - Undulating wave distortion
  - Breathing pulse (scale oscillation)
  - Skeleton trace (progressive reveal)
  - Wobble distortion
  - Mirror kaleidoscope

Output:
  sigil_animation.mp4   (1080p, 30fps, ~7 minutes)
  youtube_metadata.txt  (title, description, chapter timestamps)
"""

import cv2
import numpy as np
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    import imageio.v3 as iio
    from imageio_ffmpeg import get_ffmpeg_exe
    FFMPEG_PATH = get_ffmpeg_exe()
    print(f"Using ffmpeg from imageio-ffmpeg: {FFMPEG_PATH}")
except ImportError:
    print("ERROR: imageio-ffmpeg not installed. Run: pip install imageio-ffmpeg")
    sys.exit(1)

OUTDIR = Path(r"C:\Users\PC\Downloads\goetia_analysis")
SIGIL_DIR = OUTDIR / "extracted_sigils"

# Video parameters
WIDTH, HEIGHT = 1920, 1080
FPS = 30
SIGIL_DISPLAY_SIZE = 400  # sigil rendered at this size
BG_COLOR = (13, 17, 23)   # match dashboard dark theme
TITLE_DURATION = 3.0       # seconds for title card
SIGIL_DURATION = 5.5       # seconds per sigil
OUTRO_DURATION = 3.0       # seconds for outro

# Colors
RANK_COLORS = {
    'King': (255, 215, 0), 'Duke': (65, 105, 225), 'Prince': (147, 112, 219),
    'Marquis': (46, 139, 87), 'Earl': (220, 20, 60), 'President': (255, 140, 0),
    'Knight': (128, 128, 128)
}
CLUSTER_COLORS = {
    1: (230, 126, 34), 2: (46, 204, 113), 3: (52, 152, 219), 4: (155, 89, 182),
    5: (231, 76, 60), 6: (26, 188, 156), 7: (243, 156, 18), 8: (149, 165, 166)
}


def load_data():
    """Load all analysis data needed for animations."""
    with open(OUTDIR / "sigil_database.json") as f:
        db = json.load(f)
    return db


def create_blank_frame():
    """Create a blank 1080p frame with dark background."""
    frame = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)
    return frame


def load_sigil_image(demon):
    """Load a sigil image and prepare it for animation."""
    filename = f"sigil_{demon['id']:03d}.png"
    path = SIGIL_DIR / filename
    if not path.exists():
        return np.ones((100, 100), dtype=np.uint8) * 255

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.ones((100, 100), dtype=np.uint8) * 255

    # Resize to display size, maintaining aspect ratio
    h, w = img.shape
    scale = min(SIGIL_DISPLAY_SIZE / w, SIGIL_DISPLAY_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def put_text(frame, text, pos, font_scale=1.0, color=(230, 237, 243),
             thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX, center=False):
    """Draw text on frame, optionally centered at pos."""
    if center:
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        pos = (pos[0] - tw // 2, pos[1] + th // 2)
    cv2.putText(frame, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)


def place_sigil_on_frame(frame, sigil_bgr, cx, cy):
    """Place a BGR sigil image centered at (cx, cy) on the frame."""
    h, w = sigil_bgr.shape[:2]
    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = x1 + w
    y2 = y1 + h

    # Clip to frame bounds
    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = w - max(0, x2 - WIDTH)
    sy2 = h - max(0, y2 - HEIGHT)

    fx1 = max(0, x1)
    fy1 = max(0, y1)
    fx2 = min(WIDTH, x2)
    fy2 = min(HEIGHT, y2)

    if fx2 > fx1 and fy2 > fy1:
        frame[fy1:fy2, fx1:fx2] = sigil_bgr[sy1:sy2, sx1:sx2]


# ============================================================
# ANIMATION EFFECTS
# ============================================================

def effect_color_cycle(sigil_gray, t, period=2.0):
    """Psychedelic color cycling - map grayscale to rotating hue."""
    hue_offset = int((t / period) * 180) % 180

    # Invert so ink is bright
    inv = 255 - sigil_gray

    # Create HSV image
    h, w = inv.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = (inv.astype(np.int32) * 180 // 255 + hue_offset) % 180  # Hue
    hsv[:, :, 1] = np.where(inv > 30, 255, 0)  # Full saturation on ink
    hsv[:, :, 2] = np.where(inv > 30, 255, 20)  # Bright on ink, dark on bg

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def effect_undulate(sigil_gray, t, amplitude=8.0, freq=3.0):
    """Sinusoidal horizontal displacement creating a wave effect (vectorized)."""
    h, w = sigil_gray.shape
    ys = np.arange(h, dtype=np.float32)
    shifts = (amplitude * np.sin(2 * np.pi * (ys / h * freq + t))).astype(np.float32)

    # Build remap arrays
    xs = np.arange(w, dtype=np.float32)
    map_x = xs[np.newaxis, :] - shifts[:, np.newaxis]
    map_y = ys[:, np.newaxis] * np.ones(w, dtype=np.float32)[np.newaxis, :]

    result = cv2.remap(sigil_gray, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return result


def effect_breathe(sigil_gray, t, amplitude=0.15, period=1.5):
    """Breathing pulse - oscillating scale from center."""
    h, w = sigil_gray.shape
    scale = 1.0 + amplitude * np.sin(2 * np.pi * t / period)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    result = cv2.warpAffine(sigil_gray, M, (w, h),
                            borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return result


def effect_wobble(sigil_gray, t, amplitude=5.0):
    """Barrel/pincushion distortion oscillating in time (vectorized)."""
    h, w = sigil_gray.shape
    cx, cy = w / 2.0, h / 2.0

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy) / max(cx, cy)

    strength = amplitude * np.sin(2 * np.pi * t / 2.0)
    factor = 1.0 + strength * r * r * 0.01
    map_x = (cx + dx * factor).astype(np.float32)
    map_y = (cy + dy * factor).astype(np.float32)

    result = cv2.remap(sigil_gray, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return result


def effect_kaleidoscope(sigil_gray, t, segments=4):
    """Mirror kaleidoscope - rotate and mirror."""
    h, w = sigil_gray.shape
    angle = t * 30  # slow rotation

    # Rotate the image
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(sigil_gray, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # Create 4-fold mirror
    top_left = rotated[:h // 2, :w // 2]
    top_right = cv2.flip(top_left, 1)
    bottom_left = cv2.flip(top_left, 0)
    bottom_right = cv2.flip(top_left, -1)

    top = np.hstack([top_left, top_right[:, :w - w // 2]])
    bottom = np.hstack([bottom_left, bottom_right[:, :w - w // 2]])
    result = np.vstack([top, bottom[:h - h // 2, :]])

    return result


def effect_glitch(sigil_gray, t):
    """Random horizontal slice displacement - glitch art effect."""
    h, w = sigil_gray.shape
    result = sigil_gray.copy()

    np.random.seed(int(t * 10) % 1000)
    n_glitches = np.random.randint(3, 8)

    for _ in range(n_glitches):
        y_start = np.random.randint(0, h - 10)
        slice_h = np.random.randint(2, 15)
        shift = np.random.randint(-20, 20)

        y_end = min(y_start + slice_h, h)
        slc = result[y_start:y_end, :].copy()

        if shift > 0:
            result[y_start:y_end, shift:] = slc[:, :w - shift]
            result[y_start:y_end, :shift] = 255
        elif shift < 0:
            result[y_start:y_end, :w + shift] = slc[:, -shift:]
            result[y_start:y_end, w + shift:] = 255

    return result


# Wobble maps are now computed on the fly via vectorized effect_wobble
# No precomputation needed


# ============================================================
# FRAME GENERATORS
# ============================================================

def generate_title_frames():
    """Generate title card frames."""
    n_frames = int(TITLE_DURATION * FPS)
    frames = []

    for i in range(n_frames):
        t = i / FPS
        frame = create_blank_frame()

        # Fade in
        alpha = min(1.0, t / 1.0)

        # Title
        color = tuple(int(c * alpha) for c in (230, 237, 243))
        put_text(frame, "The 72 Seals of the Goetia",
                 (WIDTH // 2, HEIGHT // 2 - 50), 1.8, color, 3, center=True)

        # Subtitle
        sub_color = tuple(int(c * alpha) for c in (139, 148, 158))
        put_text(frame, "An Animated Analysis",
                 (WIDTH // 2, HEIGHT // 2 + 20), 1.0, sub_color, 2, center=True)

        # Attribution
        attr_color = tuple(int(c * alpha) for c in (88, 166, 255))
        put_text(frame, "github.com/t3dy/goetia-sigil-analysis",
                 (WIDTH // 2, HEIGHT // 2 + 80), 0.6, attr_color, 1, center=True)

        frames.append(frame)

    return frames


def generate_sigil_frames(demon, effect_sequence):
    """Generate animated frames for one sigil."""
    n_frames = int(SIGIL_DURATION * FPS)
    frames = []

    sigil_gray = load_sigil_image(demon)
    sh, sw = sigil_gray.shape

    # Metadata
    name = demon.get('name', '???')
    demon_id = demon.get('id', 0)
    rank = demon.get('rank', '???')
    legions = demon.get('legions', '?')
    cluster = demon.get('cluster_name', '???')
    fd = demon.get('features', {}).get('fractal_dimension', 0)
    junctions = demon.get('topology', {}).get('junctions', 0)
    holes = demon.get('topology', {}).get('holes', 0)

    rank_color = RANK_COLORS.get(rank, (180, 180, 180))
    cluster_id = demon.get('cluster_id', 1)
    cluster_color = CLUSTER_COLORS.get(cluster_id, (150, 150, 150))

    for i in range(n_frames):
        t = i / FPS
        progress = t / SIGIL_DURATION

        frame = create_blank_frame()

        # Determine which effect phase we're in
        # Phase 1 (0-1s): Fade in with name
        # Phase 2 (1-4s): Animation effects
        # Phase 3 (4-5s): Stats overlay + fade transition

        if progress < 0.18:
            # Fade in phase
            alpha = min(1.0, progress / 0.18)

            # Show sigil fading in
            sigil_bgr = cv2.cvtColor(sigil_gray, cv2.COLOR_GRAY2BGR)
            sigil_bgr = cv2.addWeighted(
                sigil_bgr, alpha,
                np.full_like(sigil_bgr, BG_COLOR), 1 - alpha, 0
            )
            place_sigil_on_frame(frame, sigil_bgr, WIDTH // 2, HEIGHT // 2)

        elif progress < 0.82:
            # Animation phase - cycle through effects
            anim_t = (progress - 0.18) / 0.64  # 0 to 1 within animation phase
            effect_idx = int(anim_t * len(effect_sequence)) % len(effect_sequence)
            effect_name = effect_sequence[effect_idx]
            local_t = t - 1.0  # time within animation

            if effect_name == 'color_cycle':
                sigil_animated = effect_color_cycle(sigil_gray, local_t, period=1.5)
            elif effect_name == 'undulate':
                sigil_undulated = effect_undulate(sigil_gray, local_t,
                                                  amplitude=6 + fd * 3, freq=2 + fd)
                sigil_animated = effect_color_cycle(sigil_undulated, local_t, period=2.0)
            elif effect_name == 'breathe':
                sigil_breathed = effect_breathe(sigil_gray, local_t, amplitude=0.12)
                sigil_animated = effect_color_cycle(sigil_breathed, local_t * 0.7, period=2.5)
            elif effect_name == 'wobble':
                sigil_wobbled = effect_wobble(sigil_gray, local_t)
                sigil_animated = effect_color_cycle(sigil_wobbled, local_t, period=1.8)
            elif effect_name == 'kaleidoscope':
                sigil_k = effect_kaleidoscope(sigil_gray, local_t)
                sigil_animated = effect_color_cycle(sigil_k, local_t * 0.5, period=3.0)
            elif effect_name == 'glitch':
                sigil_g = effect_glitch(sigil_gray, local_t)
                sigil_animated = effect_color_cycle(sigil_g, local_t, period=1.2)
            else:
                sigil_animated = effect_color_cycle(sigil_gray, local_t)

            place_sigil_on_frame(frame, sigil_animated, WIDTH // 2, HEIGHT // 2)

        else:
            # Stats overlay phase
            alpha = max(0.0, 1.0 - (progress - 0.82) / 0.18)

            sigil_bgr = effect_color_cycle(sigil_gray, t, period=3.0)
            sigil_bgr = cv2.addWeighted(
                sigil_bgr, alpha,
                np.full_like(sigil_bgr, BG_COLOR), 1 - alpha, 0
            )
            place_sigil_on_frame(frame, sigil_bgr, WIDTH // 2, HEIGHT // 2)

            # Stats text
            stats_alpha = min(1.0, (progress - 0.82) / 0.1) * alpha
            stats_color = tuple(int(c * stats_alpha) for c in (139, 148, 158))
            put_text(frame, f"FD={fd:.2f}  Junctions={junctions}  Holes={holes}",
                     (WIDTH // 2, HEIGHT // 2 + SIGIL_DISPLAY_SIZE // 2 + 70),
                     0.6, stats_color, 1, center=True)
            cl_color = tuple(int(c * stats_alpha) for c in cluster_color)
            put_text(frame, cluster,
                     (WIDTH // 2, HEIGHT // 2 + SIGIL_DISPLAY_SIZE // 2 + 100),
                     0.6, cl_color, 1, center=True)

        # Always show name/rank (with fade in/out at edges)
        text_alpha = min(1.0, progress / 0.1) * min(1.0, (1.0 - progress) / 0.1)
        name_color = tuple(int(c * text_alpha) for c in (230, 237, 243))
        rank_c = tuple(int(c * text_alpha) for c in rank_color)

        # Name and number at top
        put_text(frame, f"#{demon_id}", (WIDTH // 2 - 200, 80),
                 2.0, name_color, 3, center=False)
        put_text(frame, name, (WIDTH // 2 - 80, 80),
                 2.0, name_color, 3, center=False)

        # Rank badge
        put_text(frame, rank, (WIDTH // 2 + 300, 80),
                 1.0, rank_c, 2, center=False)

        # Legions
        leg_color = tuple(int(c * text_alpha) for c in (139, 148, 158))
        put_text(frame, f"{legions} legions", (WIDTH // 2, 120),
                 0.6, leg_color, 1, center=True)

        # Progress bar at bottom
        bar_y = HEIGHT - 20
        bar_w = int(WIDTH * progress)
        cv2.rectangle(frame, (0, bar_y), (bar_w, HEIGHT), rank_color, -1)

        frames.append(frame)

    return frames


def generate_outro_frames():
    """Generate outro/credits frames."""
    n_frames = int(OUTRO_DURATION * FPS)
    frames = []

    for i in range(n_frames):
        t = i / FPS
        alpha = min(1.0, t / 1.0) * min(1.0, (OUTRO_DURATION - t) / 0.5)
        frame = create_blank_frame()

        color = tuple(int(c * alpha) for c in (230, 237, 243))
        sub_color = tuple(int(c * alpha) for c in (139, 148, 158))
        accent = tuple(int(c * alpha) for c in (88, 166, 255))

        put_text(frame, "72 Seals Analyzed", (WIDTH // 2, HEIGHT // 2 - 60),
                 1.5, color, 3, center=True)
        put_text(frame, "8 Sigil Families Discovered",
                 (WIDTH // 2, HEIGHT // 2), 1.0, sub_color, 2, center=True)
        put_text(frame, "github.com/t3dy/goetia-sigil-analysis",
                 (WIDTH // 2, HEIGHT // 2 + 60), 0.7, accent, 1, center=True)

        frames.append(frame)

    return frames


def generate_youtube_metadata(db):
    """Generate YouTube metadata file with chapter timestamps."""
    lines = []
    lines.append("YOUTUBE VIDEO METADATA")
    lines.append("=" * 60)
    lines.append("")
    lines.append("TITLE: The 72 Seals of the Goetia - Animated Analysis")
    lines.append("")
    lines.append("DESCRIPTION:")
    lines.append("-" * 40)
    lines.append("An animated visualization of all 72 Goetic sigils from the Lesser Key of Solomon,")
    lines.append("analyzed using computer vision, graph theory, and statistical methods.")
    lines.append("")
    lines.append("Each seal is animated with psychedelic color effects and distortions,")
    lines.append("revealing the hidden structure of these centuries-old mystical symbols.")
    lines.append("")
    lines.append("8 distinct sigil families were discovered through hierarchical clustering.")
    lines.append("No correlation was found between a demon's rank and its sigil's complexity.")
    lines.append("")
    lines.append("Full analysis: https://t3dy.github.io/goetia-sigil-analysis/")
    lines.append("Source code: https://github.com/t3dy/goetia-sigil-analysis")
    lines.append("")
    lines.append("CHAPTERS:")
    lines.append("-" * 40)

    current_time = TITLE_DURATION
    for d in sorted(db, key=lambda x: x['id']):
        minutes = int(current_time) // 60
        seconds = int(current_time) % 60
        lines.append(f"{minutes:02d}:{seconds:02d} #{d['id']} {d['name']} ({d.get('rank', '?')})")
        current_time += SIGIL_DURATION

    lines.append("")
    lines.append("TAGS:")
    lines.append("-" * 40)
    lines.append("goetia, lesser key of solomon, demon sigils, occult, esoteric,")
    lines.append("computer vision, data visualization, image analysis, animation,")
    lines.append("72 demons, ceremonial magic, sigil analysis, graph theory")
    lines.append("")

    lines.append("DEMON LIST:")
    lines.append("-" * 40)
    for d in sorted(db, key=lambda x: x['id']):
        abilities = ", ".join(d.get('ability_categories', []))
        lines.append(f"#{d['id']} {d['name']} - {d.get('rank', '?')} - "
                     f"{d.get('legions', '?')} legions - {abilities}")

    meta_path = OUTDIR / "youtube_metadata.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"YouTube metadata saved to {meta_path}")


# ============================================================
# MAIN VIDEO ASSEMBLY
# ============================================================

def main():
    print("=" * 60)
    print("GOETIC SIGIL ANIMATION VIDEO GENERATOR")
    print("=" * 60)

    print("\nLoading data...")
    db = load_data()
    db_sorted = sorted(db, key=lambda x: x['id'])
    print(f"Loaded {len(db_sorted)} demons")

    # Define effect sequences for variety
    effect_sets = [
        ['color_cycle', 'undulate'],
        ['undulate', 'breathe'],
        ['color_cycle', 'wobble'],
        ['breathe', 'glitch'],
        ['kaleidoscope', 'color_cycle'],
        ['glitch', 'undulate'],
        ['wobble', 'breathe'],
        ['color_cycle', 'kaleidoscope'],
    ]

    # Generate YouTube metadata
    print("\nGenerating YouTube metadata...")
    generate_youtube_metadata(db_sorted)

    # Setup video writer using imageio
    output_path = str(OUTDIR / "sigil_animation.mp4")

    import subprocess
    import tempfile
    import os

    # Strategy: write raw frames to a temp file, then encode with ffmpeg
    # This avoids pipe issues and ensures proper finalization
    raw_path = os.path.join(tempfile.gettempdir(), "sigil_frames.raw")

    total_frames = 0
    start_time = time.time()

    print(f"\nRendering frames to temp file...")

    with open(raw_path, 'wb') as raw_file:
        def write_frames(frames):
            nonlocal total_frames
            for frame in frames:
                raw_file.write(frame.tobytes())
                total_frames += 1

        # Title card
        print("  Rendering title card...")
        write_frames(generate_title_frames())
        print(f"  Title: {total_frames} frames")

        # Each sigil
        for i, demon in enumerate(db_sorted):
            effect_seq = effect_sets[i % len(effect_sets)]
            sys.stdout.write(f"  #{demon['id']:3d} {demon['name']:20s} "
                           f"effects={effect_seq}")
            sys.stdout.flush()

            t0 = time.time()
            frames = generate_sigil_frames(demon, effect_seq)
            write_frames(frames)

            elapsed = time.time() - t0
            print(f"  ({elapsed:.1f}s, {total_frames} total)")

            # Progress update every 10 sigils
            if (i + 1) % 10 == 0:
                total_elapsed = time.time() - start_time
                est_remaining = total_elapsed / (i + 1) * (len(db_sorted) - i - 1)
                print(f"  --- Progress: {i+1}/{len(db_sorted)} sigils, "
                      f"{total_elapsed:.0f}s elapsed, ~{est_remaining:.0f}s remaining ---")

        # Outro
        print("  Rendering outro...")
        write_frames(generate_outro_frames())

    render_time = time.time() - start_time
    print(f"\nAll {total_frames} frames rendered in {render_time:.0f}s")

    # Now encode with ffmpeg
    print(f"\nEncoding MP4 with ffmpeg...")
    encode_start = time.time()

    ffmpeg_cmd = [
        FFMPEG_PATH,
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{WIDTH}x{HEIGHT}',
        '-pix_fmt', 'bgr24',
        '-r', str(FPS),
        '-i', raw_path,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '25',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path
    ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr[-300:]}")
    else:
        print(f"Encoding took {time.time() - encode_start:.0f}s")

    # Clean up temp file
    try:
        os.remove(raw_path)
        print(f"Cleaned up temp file")
    except:
        pass

    total_elapsed = time.time() - start_time
    total_duration = total_frames / FPS

    print(f"\n{'=' * 60}")
    print(f"VIDEO COMPLETE!")
    print(f"  Output: {output_path}")
    print(f"  Frames: {total_frames}")
    print(f"  Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"  Render time: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
