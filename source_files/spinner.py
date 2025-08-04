# spinner.py
#
# Contains the logic for rendering a complete, multi-layered spinner object.
# This version adds pre-computation and caching to improve performance.

from PIL import Image, ImageOps
import config_generator as cfg

# --- NEW: Cache for pre-rendered spinner frames ---
# This will store { 'asset_name': [frame_0, frame_1, ...], ... }
spinner_cache = {}


def _precompute_spinner_frames(asset_name, asset_image):
    """
    Rotates a spinner layer 360 degrees and caches each frame.
    This is a one-time cost per asset to speed up subsequent rendering.
    """
    global spinner_cache
    if asset_name in spinner_cache:
        return  # Already cached

    print(f"[INFO] Pre-computing and caching 360 frames for '{asset_name}'...")
    frames = []
    for i in range(360):
        # Use nearest neighbor resampling for speed, as quality loss is minimal for 1-degree steps
        rotated_image = asset_image.rotate(i, expand=True, resample=Image.Resampling.NEAREST)
        frames.append(rotated_image)
    spinner_cache[asset_name] = frames
    print(f"[INFO] Caching complete for '{asset_name}'.")


def render_spinner(frame_image, assets, obj, difficulty, annotations):
    """
    Renders a complete spinner, using a cache of pre-rendered rotations
    to significantly improve performance.
    """
    global spinner_cache
    current_time_ms = obj['current_time_ms']
    start_time = obj['time']
    end_time = obj.get('end_time', start_time + 1000)
    duration = end_time - start_time

    if not (start_time <= current_time_ms <= end_time):
        return

    # --- Get Assets ---
    spinner_layers = {
        'spinner-glow': 0.1,
        'spinner-middle': -0.3,
        'spinner-circle': 0.5,
        'spinner-top': -0.8
    }
    spinner_ac = assets.get('spinner-approachcircle')
    spinner_bg = assets.get('spinner-background')
    center_x, center_y = cfg.OUTPUT_RESOLUTION[0] // 2, cfg.OUTPUT_RESOLUTION[1] // 2

    annotations.append(f"{cfg.CLASS_MAP['spinner']} 0.5 0.5 1.0 1.0")

    # --- Render Static Background ---
    if spinner_bg:
        scaled_bg = spinner_bg.resize(cfg.OUTPUT_RESOLUTION, Image.Resampling.LANCZOS)
        frame_image.paste(scaled_bg, (0, 0), scaled_bg)

    # --- Pre-compute and Render Rotating Layers ---
    for name, speed_multiplier in spinner_layers.items():
        asset = assets.get(name)
        if not asset:
            continue

        # If this layer isn't cached yet, pre-compute its rotation frames
        if name not in spinner_cache:
            _precompute_spinner_frames(name, asset)

        # Calculate the angle and get the pre-rotated frame from the cache
        angle = int((current_time_ms * speed_multiplier) % 360)
        # Ensure angle is positive for list indexing
        if angle < 0: angle += 360

        cached_frames = spinner_cache.get(name)
        if cached_frames and len(cached_frames) == 360:
            rotated_layer = cached_frames[angle]
            paste_pos = (center_x - rotated_layer.width // 2, center_y - rotated_layer.height // 2)
            frame_image.paste(rotated_layer, paste_pos, rotated_layer)

    # --- Render Approach Circle ---
    if spinner_ac:
        time_since_start = current_time_ms - start_time
        progress = max(0, min(1, time_since_start / duration if duration > 0 else 1))
        scale = 1.0 - progress
        if scale > 0:
            ac_size = (int(spinner_ac.width * scale), int(spinner_ac.height * scale))
            if ac_size[0] > 0 and ac_size[1] > 0:
                ac_resized = spinner_ac.resize(ac_size, Image.Resampling.LANCZOS)
                ac_pos = (center_x - ac_resized.width // 2, center_y - ac_resized.height // 2)
                frame_image.paste(ac_resized, ac_pos, ac_resized)