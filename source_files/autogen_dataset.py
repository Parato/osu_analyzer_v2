# autogen_dataset.py

# This is the main script to run the automated dataset generator.
# MODIFIED FOR VERTEX AI:
# - Added google-cloud-storage library to read/write from/to GCS.
# - Image and label files are now saved directly to a GCS bucket.
# - Local file operations are replaced with GCS-compatible logic.

import cv2
import numpy as np
import argparse
import os
import random
from osrparse import Replay
from tqdm import tqdm
import math
import sys
from PIL import Image, ImageFont
import bezier
import io

# --- Vertex AI Imports ---
from google.cloud import storage
from urllib.parse import urlparse

# --- Local Imports ---
from utils import print_status as original_print_status
from osu_parser import parse_beatmap
from skin_loader import load_skin_assets
from renderer import render_frame, reset_renderer_state, get_cs_radius, UI_SCALE
import config_generator as cfg
from game_simulation import GameSimulation

# --- Global flag for silent mode ---
SILENT_MODE = False
DT_RATE = 1.5

# --- GCS Client ---
# Initialized globally to be reused.
storage_client = None


def get_gcs_client():
    """Initializes and returns a global GCS client."""
    global storage_client
    if storage_client is None:
        storage_client = storage.Client()
    return storage_client


def parse_gcs_path(gcs_path):
    """Parses a GCS path into bucket name and blob name."""
    parsed = urlparse(gcs_path)
    if not parsed.scheme == 'gs':
        raise ValueError("Path must be a GCS path (gs://...).")
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip('/')
    return bucket_name, blob_name


def read_gcs_file(gcs_path):
    """Reads a file from GCS and returns its content as bytes."""
    client = get_gcs_client()
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()


def upload_to_gcs(data, gcs_path, content_type='application/octet-stream'):
    """Uploads data (bytes) to a GCS path."""
    client = get_gcs_client()
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)


def gcs_path_exists(gcs_path):
    """Checks if a file or folder exists in GCS."""
    client = get_gcs_client()
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    bucket = client.bucket(bucket_name)
    # Check for file
    if storage.Blob(bucket=bucket, name=blob_name).exists(client):
        return True
    # Check for folder (an object with that prefix)
    blobs = client.list_blobs(bucket_name, prefix=blob_name, delimiter='/', max_results=1)
    if len(list(blobs)) > 0:
        return True
    return False


def print_status(message, level="INFO"):
    """Prints a formatted status message, unless in silent mode."""
    if not SILENT_MODE:
        original_print_status(message, level)


# --- The rest of the autogen_dataset.py script remains largely the same ---
# ... (all your existing functions like ReplayKeys, get_ar_ms, apply_stack_leniency, etc.)
# The key change is in the main loop where files are saved.

class ReplayKeys:
    M1 = 1
    M2 = 2
    K1 = 4
    K2 = 8
    SMOKE = 16


def get_ar_ms(ar):
    """Converts Approach Rate (AR) to milliseconds."""
    if ar < 5:
        return 1200 + 600 * (5 - ar) / 5
    elif ar == 5:
        return 1200
    else:
        return 1200 - 750 * (ar - 5) / 5


def apply_stack_leniency(hit_objects, difficulty):
    print_status("Applying stack leniency and HR visual modifications...")

    cs = difficulty.get('CircleSize', 4)
    ar = difficulty.get('ApproachRate', 9)
    stack_leniency = difficulty.get('StackLeniency', 0.7)

    stack_offset_osu_px = ((109 - 9 * cs) / 2) * 0.1
    stack_time_window = get_ar_ms(ar) * stack_leniency

    for i in range(len(hit_objects)):
        current_obj = hit_objects[i]
        current_obj['stack_count'] = 0
        if current_obj.get('is_spinner'): continue

        for j in range(i - 1, -1, -1):
            prev_obj = hit_objects[j]
            if prev_obj.get('is_spinner'): continue
            time_diff = current_obj['time'] - prev_obj['time']
            if time_diff > stack_time_window: break

            if prev_obj.get('is_slider'):
                slider_end_time = prev_obj['time'] + prev_obj.get('slider_duration', 0)
                if slider_end_time >= current_obj['time']: continue

            prev_end_x, prev_end_y = prev_obj['x'], prev_obj['y']
            if prev_obj.get('is_slider') and prev_obj.get('curve'):
                end_nodes = prev_obj['curve'].nodes[:, -1]
                prev_end_x, prev_end_y = end_nodes[0], end_nodes[1]

            dist_sq = (current_obj['x'] - prev_end_x) ** 2 + (current_obj['y'] - prev_end_y) ** 2
            if dist_sq < 1:
                current_obj['stack_count'] = prev_obj['stack_count'] + 1
                break

    for obj in hit_objects:
        offset = obj['stack_count'] * stack_offset_osu_px
        obj['render_x'] = obj['x'] + offset
        obj['render_y'] = obj['y'] + offset

    if difficulty.get('is_hr_applied'):
        for obj in hit_objects:
            obj['render_y'] = 384 - obj['render_y']
            if obj.get('is_slider') and obj.get('curve'):
                nodes = obj['curve'].nodes.copy()
                nodes[1, :] = 384 - nodes[1, :]
                obj['curve'] = bezier.Curve(nodes, degree=len(nodes[0]) - 1)

    print_status("Stack leniency and HR modifications applied.")


def calculate_slider_durations(hit_objects, difficulty, timing_points):
    slider_multiplier = difficulty.get('SliderMultiplier', 1.4)
    first_uninherited_point = None
    for point in timing_points:
        if point['uninherited']:
            first_uninherited_point = point
            break
    if first_uninherited_point is None:
        print_status("No uninherited timing points found. Cannot calculate slider durations.", level="ERROR")
        return
    current_beat_length = first_uninherited_point['beat_length']
    current_slider_sv = 1.0
    timing_point_idx = 0
    for obj in hit_objects:
        while timing_point_idx + 1 < len(timing_points) and timing_points[timing_point_idx + 1]['time'] <= obj['time']:
            timing_point_idx += 1
            point = timing_points[timing_point_idx]
            if point['uninherited']:
                current_beat_length = point['beat_length']
                current_slider_sv = 1.0
            elif point['beat_length'] < 0:
                current_slider_sv = (-100.0 / point['beat_length'])
        if obj.get('is_slider'):
            pixels_per_beat = slider_multiplier * 100 * current_slider_sv
            if pixels_per_beat == 0: obj['slider_duration'] = 0; continue
            beats_for_slider = obj['length'] / pixels_per_beat
            obj['slider_duration'] = beats_for_slider * current_beat_length * obj['slides']
        else:
            obj['slider_duration'] = 0


def simulate_play(hit_objects, replay_events, difficulty, timing_windows):
    if not replay_events:
        return
    print_status("Simulating play to determine hit timings for rendering...")

    objects_to_hit = []
    for obj in hit_objects:
        if obj.get('is_spinner'):
            obj['hit_time'] = obj['time']
            obj['hit_result'] = '300'
        elif not obj.get('hit_time'):
            objects_to_hit.append(obj)

    event_idx = 0
    for h_obj in objects_to_hit:
        start_time = h_obj['time'] - timing_windows['50']
        end_time = h_obj['time'] + timing_windows['50']

        found_hit = False
        for i in range(event_idx, len(replay_events)):
            event = replay_events[i]
            if event.time_ms > end_time:
                break
            if event.time_ms >= start_time:
                prev_keys = replay_events[i - 1].keys if i > 0 else 0
                keys_pressed_this_event = event.keys & ~prev_keys
                if keys_pressed_this_event > 0:
                    dist = math.hypot(event.x - h_obj['x'], event.y - h_obj['y'])
                    cs_radius_px = (109 - 9 * difficulty.get('CircleSize', 4)) / 2
                    if dist <= cs_radius_px:
                        h_obj['hit_time'] = event.time_ms
                        time_diff = abs(event.time_ms - h_obj['time'])
                        if time_diff <= timing_windows['300']:
                            h_obj['hit_result'] = '300'
                        elif time_diff <= timing_windows['100']:
                            h_obj['hit_result'] = '100'
                        else:
                            h_obj['hit_result'] = '50'
                        event_idx = i + 1
                        found_hit = True
                        break
        if not found_hit:
            while event_idx < len(replay_events) and replay_events[event_idx].time_ms <= end_time:
                event_idx += 1


def get_hp_for_frame(current_time_ms, life_bar_graph):
    if not life_bar_graph:
        return 1.0
    prev_event = life_bar_graph[0]
    next_event = None
    for event in life_bar_graph[1:]:
        if event.time >= current_time_ms:
            next_event = event
            break
        prev_event = event
    if next_event is None:
        return prev_event.life
    time_diff = next_event.time - prev_event.time
    if time_diff == 0:
        return prev_event.life
    progress = (current_time_ms - prev_event.time) / time_diff
    interp_life = prev_event.life + (next_event.life - prev_event.life) * progress
    return max(0.0, min(1.0, interp_life))


def get_key_states_for_frames(replay_events, start_frame, end_frame, frame_rate):
    if not replay_events: return {}
    key_states = {}
    event_idx = 0
    start_time_ms = start_frame * 1000 / frame_rate
    while event_idx + 1 < len(replay_events) and replay_events[event_idx + 1].time_ms <= start_time_ms:
        event_idx += 1

    for i in range(start_frame, end_frame):
        current_time_ms = i * 1000 / frame_rate
        while event_idx + 1 < len(replay_events) and replay_events[event_idx + 1].time_ms <= current_time_ms:
            event_idx += 1
        event = replay_events[event_idx]
        keys_bitmask = event.keys
        key_states[i] = {
            'm1': bool(keys_bitmask & ReplayKeys.M1), 'm2': bool(keys_bitmask & ReplayKeys.M2),
            'k1': bool(keys_bitmask & ReplayKeys.K1), 'k2': bool(keys_bitmask & ReplayKeys.K2)
        }
    return key_states


def get_cursor_positions_for_frames(replay_events, start_frame, end_frame, frame_rate):
    if not replay_events: return {}

    positions = {}
    event_idx = 0
    start_time_ms = start_frame * 1000 / frame_rate
    while event_idx + 1 < len(replay_events) and replay_events[event_idx + 1].time_ms <= start_time_ms:
        event_idx += 1

    prev_event = replay_events[event_idx]
    next_event = replay_events[event_idx + 1] if event_idx + 1 < len(replay_events) else prev_event

    for i in range(start_frame, end_frame):
        current_time_ms = i * 1000 / frame_rate

        while next_event and current_time_ms > next_event.time_ms:
            prev_event = next_event
            event_idx += 1
            next_event = replay_events[event_idx + 1] if event_idx + 1 < len(replay_events) else None
            if not next_event:
                break

        if not next_event:
            interp_x, interp_y = prev_event.x, prev_event.y
        else:
            time_diff = next_event.time_ms - prev_event.time_ms
            if time_diff == 0:
                interp_x, interp_y = prev_event.x, prev_event.y
            else:
                progress = (current_time_ms - prev_event.time_ms) / time_diff
                interp_x = prev_event.x + (next_event.x - prev_event.x) * progress
                interp_y = prev_event.y + (next_event.y - prev_event.y) * progress

        positions[i] = (interp_x, interp_y)
    return positions


def get_opaque_bbox_dimensions(image):
    if not image or image.mode != 'RGBA':
        return None, None

    alpha = image.getchannel('A')
    OPAQUE_THRESHOLD = 150
    opaque_points = np.argwhere(np.array(alpha) > OPAQUE_THRESHOLD)

    if opaque_points.size == 0:
        return None, None

    min_y, min_x = opaque_points.min(axis=0)
    max_y, max_x = opaque_points.max(axis=0)

    width = max_x - min_x + 1
    height = max_y - min_y + 1
    return width, height


def pre_render_assets(assets, difficulty):
    print_status("Pre-rendering and caching assets...", level="INFO")
    pre_rendered = {}
    playfield_scale = cfg.PLAYFIELD_HEIGHT / 384.0
    cs_radius = get_cs_radius(difficulty.get('CircleSize', 4))
    circle_pixel_radius = int(cs_radius * playfield_scale)

    base_hc_asset = assets.get('hitcircleoverlay') or assets.get('hitcircle')
    if base_hc_asset:
        hc_size = (circle_pixel_radius * 2, circle_pixel_radius * 2)
    else:
        hc_size = (0, 0)

    gameplay_assets_to_resize = [
        'hitcircle', 'hitcircleoverlay', 'approachcircle', 'reversearrow',
        'hit300', 'sliderfollowcircle', 'hit0'
    ]
    is_digit_fallback = assets.get('is_digit_fallback', False)

    for name in gameplay_assets_to_resize:
        if name in assets and assets[name] and hc_size[0] > 0 and hc_size[1] > 0:
            asset_img = assets[name]
            if name == 'hit0':
                bbox = asset_img.getbbox()
                if bbox:
                    asset_img = asset_img.crop(bbox)
                if asset_img.width == 0 or asset_img.height == 0:
                    continue

            if name == 'hit300' or name == 'hit0':
                burst_size = (int(circle_pixel_radius * 1.3), int(circle_pixel_radius * 1.3))
                pre_rendered[name] = asset_img.resize(burst_size, Image.Resampling.LANCZOS)
            elif name == 'sliderfollowcircle':
                follow_diameter = int(circle_pixel_radius * 2.4 * 2)
                pre_rendered[name] = asset_img.resize((follow_diameter, follow_diameter), Image.Resampling.LANCZOS)

            elif name == 'hitcircleoverlay' and is_digit_fallback:
                print_status("Applying special proportional scaling for digit-based hitcircleoverlay.", level="INFO")
                digit_canvas = Image.new('RGBA', hc_size, (0, 0, 0, 0))
                digit_copy = asset_img.copy()
                digit_copy.thumbnail(hc_size, Image.Resampling.LANCZOS)
                paste_x = (hc_size[0] - digit_copy.width) // 2
                paste_y = (hc_size[1] - digit_copy.height) // 2
                digit_canvas.paste(digit_copy, (paste_x, paste_y), digit_copy)
                pre_rendered[name] = digit_canvas
            else:
                pre_rendered[name] = asset_img.resize(hc_size, Image.Resampling.LANCZOS)

    if 'hitcircleoverlay' in pre_rendered:
        composite_canvas = Image.new('RGBA', hc_size, (0, 0, 0, 0))
        overlay = pre_rendered['hitcircleoverlay']
        circle = pre_rendered.get('hitcircle')
        composite_canvas.paste(overlay, (0, 0), overlay)
        if circle:
            composite_canvas = Image.alpha_composite(composite_canvas, circle)
        pre_rendered['hitcircle_opaque_dims'] = get_opaque_bbox_dimensions(composite_canvas)

    if 'hit0' in pre_rendered:
        pre_rendered['hit0_opaque_dims'] = get_opaque_bbox_dimensions(pre_rendered['hit0'])

    if 'slider-tick' in assets and assets['slider-tick']:
        tick_size = (int(circle_pixel_radius * 0.7), int(circle_pixel_radius * 0.7))
        if tick_size[0] > 0 and tick_size[1] > 0:
            pre_rendered['slider-tick'] = assets['slider-tick'].resize(tick_size, Image.Resampling.LANCZOS)

    spinner_layers_to_crop = ['spinner-circle', 'spinner-glow', 'spinner-middle', 'spinner-top']
    for name in spinner_layers_to_crop:
        if name in assets and assets[name]:
            asset_img = assets[name]
            bbox = asset_img.getbbox()
            if bbox:
                pre_rendered[name] = asset_img.crop(bbox)
            else:
                pre_rendered[name] = asset_img

    if 'cursor' in assets and assets['cursor']:
        cursor_img = assets['cursor'].copy()
        bbox = cursor_img.getbbox()
        if bbox:
            cursor_img = cursor_img.crop(bbox)

        if cursor_img.width > 0 and cursor_img.height > 0:
            opaque_w, opaque_h = get_opaque_bbox_dimensions(cursor_img)
            orig_opaque_w = opaque_w if opaque_w and opaque_w > 0 else cursor_img.width
            orig_opaque_h = opaque_h if opaque_h and opaque_h > 0 else cursor_img.height
            orig_opaque_max_dim = max(orig_opaque_w, orig_opaque_h)

            target_opaque_dim = max(15, int(circle_pixel_radius * 0.75))

            scale_ratio = target_opaque_dim / orig_opaque_max_dim if orig_opaque_max_dim > 0 else 0

            final_cursor_w = max(1, int(cursor_img.width * scale_ratio))
            final_cursor_h = max(1, int(cursor_img.height * scale_ratio))
            final_scaled_cursor = cursor_img.resize((final_cursor_w, final_cursor_h), Image.Resampling.LANCZOS)
            pre_rendered['cursor'] = final_scaled_cursor

            pre_rendered['cursor_opaque_dims'] = get_opaque_bbox_dimensions(final_scaled_cursor)

            if 'cursortrail' in assets and assets['cursortrail']:
                trail_img = assets['cursortrail'].copy()
                trail_bbox = trail_img.getbbox()
                if trail_bbox:
                    trail_img = trail_img.crop(trail_bbox)

                if trail_img.width > 0 and trail_img.height > 0:
                    final_trail_w = max(1, int(trail_img.width * scale_ratio))
                    final_trail_h = max(1, int(trail_img.height * scale_ratio))
                    pre_rendered['cursortrail'] = trail_img.resize((final_trail_w, final_trail_h),
                                                                   Image.Resampling.LANCZOS)

    ui_assets_to_resize = []
    for i in range(10): ui_assets_to_resize.extend([f'default-{i}', f'score-{i}', f'combo-{i}'])
    ui_assets_to_resize.extend(['score-dot', 'score-percent', 'score-x', 'combo-x'])

    for name in ui_assets_to_resize:
        if name in assets and assets[name]:
            img = assets[name]
            new_size = (max(1, int(img.width * UI_SCALE)), max(1, int(img.height * UI_SCALE)))
            if new_size[0] > 0 and new_size[1] > 0:
                pre_rendered[f"{name}_ui"] = img.resize(new_size, Image.Resampling.LANCZOS)

    for key, value in assets.items():
        if key not in pre_rendered:
            pre_rendered[key] = value

    print_status(f"Finished pre-rendering {len(pre_rendered)} asset variants.", level="INFO")
    return pre_rendered


def main():
    global SILENT_MODE
    parser = argparse.ArgumentParser(description="Automated Dataset Generator for osu!")
    # MODIFIED: Input paths can now be local or GCS paths
    parser.add_argument("beatmap_path", help="Path to the .osu beatmap file (local or gs://).")
    parser.add_argument("skin_path", help="Path to the skin folder (local or gs://).")
    parser.add_argument("--replay_path", help="Optional path to an .osr replay file (local or gs://).", default=None)
    parser.add_argument("--background-path", help="Optional path to a background image (local or gs://).", default=None)
    # MODIFIED: Output directory must be a GCS path
    parser.add_argument("--output-dir", help="GCS directory to save the generated dataset (gs://...).",
                        default="gs://your-bucket/generated_dataset")

    parser.add_argument("--background-opacity", type=float, default=0.1,
                        help="Opacity for the background image (0.0 to 1.0).")
    parser.add_argument("--start-time", type=int, default=0,
                        help="Start time in milliseconds for the generated snippet.")
    parser.add_argument("--duration", type=int, default=10000,
                        help="Duration in milliseconds for the generated snippet.")
    parser.add_argument("--hd", action="store_true", help="Enable Hidden (HD) mod simulation.")
    parser.add_argument("--hr", action="store_true", help="Enable Hard Rock (HR) mod simulation.")
    parser.add_argument("--dt", action="store_true", help="Enable Double Time (DT) mod simulation.")
    parser.add_argument("--full-replay-sim", action="store_true",
                        help="Force the use of full replay simulation for hits, keys, and HP.")
    parser.add_argument("--val-split", type=float, default=0.0, help="Ratio for validation set.")
    parser.add_argument("--filename-prefix", help="Prefix for output filenames to ensure uniqueness.", default="")
    parser.add_argument("--reporter", action="store_true", help="Enable reporter mode for master pipeline.")
    parser.add_argument("--cursor-only", action="store_true",
                        help="Generate a clip with only the cursor visible (no hit objects).")
    parser.add_argument("--no-cursor", action="store_true", help="Generate a clip with no cursor rendered or labeled.")
    parser.add_argument("--no-misses", action="store_true",
                        help="Assume perfect play, preventing any misses from being simulated or rendered.")

    args = parser.parse_args()

    # --- Start of main logic ---
    if args.hr and (args.dt or args.hd):
        print_status("HR cannot be combined with DT or HD. Disabling HR for this run.", level="WARN")
        args.hr = False

    if args.reporter:
        SILENT_MODE = True

    if args.cursor_only and not args.replay_path:
        print_status("The --cursor-only flag requires a replay file (--replay_path) to be provided. Exiting.",
                     level="ERROR")
        return

    # MODIFIED: Load beatmap from local or GCS path
    if args.beatmap_path.startswith('gs://'):
        beatmap_content = read_gcs_file(args.beatmap_path).decode('utf-8')
        hit_objects, difficulty, timing_points, beatmap_combo_colors, timing_windows = parse_beatmap(
            args.beatmap_path, content=beatmap_content, apply_hr=args.hr, apply_dt=args.dt
        )
    else:
        hit_objects, difficulty, timing_points, beatmap_combo_colors, timing_windows = parse_beatmap(
            args.beatmap_path, apply_hr=args.hr, apply_dt=args.dt
        )

    if not hit_objects: return

    if args.hr:
        difficulty['is_hr_applied'] = True

    # MODIFIED: Load skin from local or GCS path
    skin_assets, skin_combo_colors = load_skin_assets(args.skin_path)
    if not skin_assets: return

    print_status("Pre-loading fonts...", level="INFO")
    try:
        main_font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        main_font = ImageFont.load_default()
        print_status("Arial not found, falling back to default PIL font.", level="WARN")

    pre_rendered_assets = pre_render_assets(skin_assets, difficulty)

    print_status("Calculating accurate slider durations...")
    calculate_slider_durations(hit_objects, difficulty, timing_points)

    apply_stack_leniency(hit_objects, difficulty)

    background_image = None
    if args.background_path:
        try:
            # MODIFIED: Load background from local or GCS path
            if args.background_path.startswith('gs://'):
                bg_bytes = read_gcs_file(args.background_path)
                background_image = Image.open(io.BytesIO(bg_bytes)).convert('RGBA')
            elif os.path.exists(args.background_path):
                background_image = Image.open(args.background_path).convert('RGBA')

            if background_image:
                print_status(f"Using background image from: {args.background_path}", level="INFO")
            else:
                print_status(f"Provided background image not found at: {args.background_path}", level="WARN")
        except Exception as e:
            print_status(f"Could not load provided background image: {e}", level="ERROR")

    # ... (rest of the logic for colors, replay parsing, etc. is mostly the same)
    if beatmap_combo_colors:
        combo_colors = beatmap_combo_colors
        print_status(f"Using {len(combo_colors)} combo colors defined in the beatmap.", level="INFO")
    else:
        combo_colors = skin_combo_colors
        print_status(f"Using {len(combo_colors)} combo colors defined in the skin (as beatmap defines none).",
                     level="INFO")

    if combo_colors:
        print_status(f"Assigning combo color indices to hit objects...")
        current_color_index = 0
        if hit_objects:
            hit_objects[0]['combo_color_index'] = current_color_index
            for i in range(1, len(hit_objects)):
                if hit_objects[i].get('new_combo', False):
                    current_color_index = (current_color_index + 1) % len(combo_colors)
                hit_objects[i]['combo_color_index'] = current_color_index
    else:
        print_status("No combo colors found in beatmap or skin. Using default tint.", level="WARN")
        for obj in hit_objects:
            obj['combo_color_index'] = 0

    replay_events = None
    life_bar_graph = None
    cursor_positions_by_frame = {}
    key_states_by_frame = {}
    is_faithful_replay = False

    if args.replay_path:
        try:
            # MODIFIED: Load replay from local or GCS path
            if args.replay_path.startswith('gs://'):
                replay_bytes = read_gcs_file(args.replay_path)
                replay = Replay.from_file(io.BytesIO(replay_bytes))
            else:
                replay = Replay.from_path(args.replay_path)

            speed_rate = DT_RATE if args.dt else 1.0
            gameplay_events = replay.replay_data[:-1]

            current_time = 0.0
            for event in gameplay_events:
                current_time += event.time_delta
                event.time_ms = current_time / speed_rate
                if args.hr:
                    event.y = 384 - event.y
            replay_events = gameplay_events

            if replay.life_bar_graph:
                for event in replay.life_bar_graph:
                    event.time = event.time / speed_rate
                life_bar_graph = replay.life_bar_graph

            print_status(f"Successfully loaded replay for player: {replay.username}")
            if args.dt: print_status("Replay timestamps have been adjusted for DT.", level="INFO")
            if args.hr: print_status("Replay cursor path has been flipped for HR.", level="INFO")

            is_modified_cs_ar = "_cs" in os.path.basename(args.beatmap_path)

            if args.full_replay_sim or (not args.hr and not is_modified_cs_ar):
                is_faithful_replay = True
                print_status("Replay is valid for full simulation (hits, keys, HP).")
                simulate_play(hit_objects, replay_events, difficulty, timing_windows)
            else:
                reason = "Cursor path only:"
                if args.hr:
                    reason += " HR mod is active."
                elif is_modified_cs_ar:
                    reason += " CS/AR have been modified."
                print_status(reason, level="WARN")

        except Exception as e:
            print_status(f"Could not load or process replay file: {e}", level="ERROR")
            replay_events = None

    start_frame = int(args.start_time / 1000 * cfg.FRAME_RATE)
    end_frame = int((args.start_time + args.duration) / 1000 * cfg.FRAME_RATE)
    num_frames_to_generate = end_frame - start_frame

    if num_frames_to_generate <= 0:
        print_status(f"Start time {args.start_time} and duration {args.duration} result in 0 frames. Exiting.",
                     level="ERROR")
        return

    if replay_events:
        print_status("Pre-calculating cursor positions and key states from replay...")
        cursor_positions_by_frame = get_cursor_positions_for_frames(replay_events, start_frame, end_frame,
                                                                    cfg.FRAME_RATE)
        if is_faithful_replay:
            key_states_by_frame = get_key_states_for_frames(replay_events, start_frame, end_frame, cfg.FRAME_RATE)

    game_sim = GameSimulation(difficulty, timing_windows, no_misses=args.no_misses)

    # MODIFIED: No need to create local directories. Paths are now GCS paths.
    output_bucket, output_prefix = parse_gcs_path(args.output_dir)
    train_images_dir = f"gs://{output_bucket}/{output_prefix}/images/train"
    train_labels_dir = f"gs://{output_bucket}/{output_prefix}/labels/train"
    val_images_dir = f"gs://{output_bucket}/{output_prefix}/images/val"
    val_labels_dir = f"gs://{output_bucket}/{output_prefix}/labels/val"

    print_status(f"Output will be saved to: {args.output_dir}")
    if args.hd: print_status("Hidden (HD) mod simulation is ENABLED.", level="WARN")
    if args.hr: print_status("Hard Rock (HR) mod simulation is ENABLED.", level="WARN")
    if args.dt: print_status("Double Time (DT) mod simulation is ENABLED.", level="WARN")
    if args.no_cursor: print_status("NO CURSOR mode is ENABLED for this clip.", level="WARN")
    if args.cursor_only: print_status("CURSOR ONLY mode is ENABLED for this clip.", level="WARN")

    reset_renderer_state()

    use_scaling_animation = random.random() < 0.8
    if use_scaling_animation and not args.hd:
        print_status("Scaling hit/fade animation is ENABLED for this clip.", level="INFO")
    else:
        print_status("Scaling hit/fade animation is DISABLED for this clip.", level="INFO")

    frame_iterator = range(start_frame, end_frame)
    if not args.reporter:
        frame_iterator = tqdm(frame_iterator, desc="Generating Snippet", leave=True, ncols=100, file=sys.stderr)

    render_objects = not args.cursor_only
    render_the_cursor = not args.no_cursor
    render_the_ui = is_faithful_replay and not args.cursor_only and not args.no_cursor

    for i, frame_num in enumerate(frame_iterator):
        if args.reporter:
            print(f"PROG {i + 1}", flush=True)

        current_time_ms = frame_num * 1000 / cfg.FRAME_RATE
        key_state = key_states_by_frame.get(frame_num, {'m1': False, 'm2': False, 'k1': False, 'k2': False})

        cursor_pos = None
        if render_the_cursor:
            cursor_pos = cursor_positions_by_frame.get(frame_num)

        sim_state = game_sim.update_state(current_time_ms, hit_objects, key_state, cursor_pos)

        if is_faithful_replay:
            sim_state['hp'] = get_hp_for_frame(current_time_ms, life_bar_graph) if life_bar_graph else 1.0
        else:
            sim_state['hp'] = 1.0

        frame_image, annotation_data = render_frame(
            frame_num, hit_objects, pre_rendered_assets, difficulty,
            combo_colors,
            cursor_pos,
            is_hd=args.hd,
            game_simulation_state=sim_state,
            render_ui=render_the_ui,
            key_state=key_state,
            background_image=background_image,
            background_opacity=args.background_opacity,
            main_font=main_font,
            use_scaling_animation=use_scaling_animation,
            render_objects=render_objects
        )

        frame_cv = cv2.cvtColor(np.array(frame_image), cv2.COLOR_RGBA2RGB)
        frame_image = Image.fromarray(frame_cv).convert('RGB')

        is_validation = random.random() < args.val_split
        images_base_dir = val_images_dir if is_validation else train_images_dir
        labels_base_dir = val_labels_dir if is_validation else train_labels_dir

        prefix_sep = f"{args.filename_prefix}_" if args.filename_prefix else ""
        base_filename = f"{prefix_sep}frame_{frame_num:06d}"

        # --- MODIFIED: Save to GCS ---
        # 1. Save image to an in-memory buffer
        img_byte_arr = io.BytesIO()
        frame_image.save(img_byte_arr, format='JPEG', quality=random.randint(90, 98))
        img_byte_arr = img_byte_arr.getvalue()

        # 2. Upload image buffer to GCS
        image_path = f"{images_base_dir}/{base_filename}.jpg"
        upload_to_gcs(img_byte_arr, image_path, content_type='image/jpeg')

        # 3. Upload label file if it exists
        if annotation_data:
            label_path = f"{labels_base_dir}/{base_filename}.txt"
            upload_to_gcs(annotation_data.encode('utf-8'), label_path, content_type='text/plain')

    if args.reporter:
        print("PROG_DONE", flush=True)

    print_status("Dataset generation complete.")


if __name__ == "__main__":
    main()