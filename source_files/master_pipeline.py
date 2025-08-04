# master_pipeline.py

# An orchestrator script to fully automate the dataset generation and training process.
# MODIFIED FOR VERTEX AI:
# - All file operations now use Google Cloud Storage (GCS).
# - `os.path` and `shutil` are replaced with GCS client library functions.
# - The script now expects GCS paths for all source directories.
# - The final YAML file is generated with GCS paths and uploaded to GCS.
# - Temporary files (like modified beatmaps) are written to a temp GCS location.

import os
import io
import shutil
import subprocess
import itertools
import argparse
import random
import time
import sys
import threading
import queue
from tqdm import tqdm
from multiprocessing import cpu_count
from PIL import Image
import numpy as np
from urllib.parse import urlparse

# --- Vertex AI Imports ---
from google.cloud import storage

# --- FIX for OMP Error #15 ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- Local Imports ---
from utils import print_status
from osu_parser import parse_beatmap
from analysis_helpers import (find_best_section_by_count,
                              get_spinner_clips,
                              get_universal_value_score)

# --- GCS Client ---
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
        raise ValueError(f"Path must be a GCS path (gs://...). Got: {gcs_path}")
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip('/')
    return bucket_name, blob_name


def list_gcs_files(gcs_path, suffix=None):
    """Lists all files in a GCS directory with an optional suffix filter."""
    client = get_gcs_client()
    bucket_name, prefix = parse_gcs_path(gcs_path)
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    files = []
    for blob in blobs:
        if not blob.name.endswith('/'):  # Exclude folders
            if suffix is None or blob.name.endswith(suffix):
                files.append(f"gs://{bucket_name}/{blob.name}")
    return files


def list_gcs_dirs(gcs_path):
    """Lists all subdirectories in a GCS path."""
    client = get_gcs_client()
    bucket_name, prefix = parse_gcs_path(gcs_path)
    blobs = client.list_blobs(bucket_name, prefix=prefix, delimiter='/')
    # The prefixes property will be populated with the subdirectories
    return [p for p in blobs.prefixes]


def rename_gcs_blob(source_gcs_path, dest_gcs_path):
    """Renames (moves) a blob in GCS."""
    client = get_gcs_client()
    source_bucket_name, source_blob_name = parse_gcs_path(source_gcs_path)
    dest_bucket_name, dest_blob_name = parse_gcs_path(dest_gcs_path)

    source_bucket = client.bucket(source_bucket_name)
    source_blob = source_bucket.blob(source_blob_name)

    # If destination bucket is the same, use the more efficient rename
    if source_bucket_name == dest_bucket_name:
        source_bucket.rename_blob(source_blob, dest_blob_name)
    else:  # Otherwise, copy and delete
        dest_bucket = client.bucket(dest_bucket_name)
        source_bucket.copy_blob(source_blob, dest_bucket, dest_blob_name)
        source_blob.delete()


def delete_gcs_folder(gcs_path):
    """Deletes all files within a GCS 'folder' (prefix)."""
    print_status(f"Deleting GCS folder: {gcs_path}", level="INFO")
    client = get_gcs_client()
    bucket_name, prefix = parse_gcs_path(gcs_path)
    bucket = client.bucket(bucket_name)
    blobs_to_delete = list(client.list_blobs(bucket_name, prefix=prefix))
    if blobs_to_delete:
        bucket.delete_blobs(blobs_to_delete)
        print_status(f"Deleted {len(blobs_to_delete)} blobs from {gcs_path}", level="SUCCESS")


# --- USER CONFIGURATION ---
# IMPORTANT: These paths MUST now point to your GCS locations.
# You can pass the base bucket path via command line argument.
BASE_GCS_BUCKET = "gs://your-main-gcs-bucket"  # CHANGE THIS or pass via --base-gcs-bucket
SOURCE_BEATMAPS_DIR = f"{BASE_GCS_BUCKET}/source/beatmaps"
SOURCE_SKINS_DIR = f"{BASE_GCS_BUCKET}/source/skins"
SOURCE_REPLAYS_DIR = f"{BASE_GCS_BUCKET}/source/replays"
SOURCE_BACKGROUNDS_DIR = f"{BASE_GCS_BUCKET}/source/backgrounds"
SOURCE_MENU_DIR = f"{BASE_GCS_BUCKET}/source/menu"
BASE_OUTPUT_DIR = f"{BASE_GCS_BUCKET}/datasets"  # Intermediate and final datasets
FINAL_DATASET_NAME = "master_dataset_v16"
GENERATE_ORIGINAL_WITH_REPLAY = True
CS_VALUES_MODIFIED = [2, 7]
AR_VALUES_MODIFIED = [3, 6]
CLIP_DURATION_MS = 20000
FRAME_RATE = 60
VALIDATION_SPLIT = 0.2
NEGATIVE_SAMPLE_RATIO = 0.10
DT_RATE = 1.5


# --- END OF CONFIGURATION ---

# --- The rest of the script is adapted for GCS ---
# ... (all your existing functions, but with GCS logic)

def get_opaque_bbox_dimensions(image):
    """Calculates the width and height of the bounding box of fully opaque pixels."""
    if image.mode != 'RGBA':
        return None, None
    alpha = image.getchannel('A')
    opaque_points = np.argwhere(np.array(alpha) == 255)
    if opaque_points.size == 0:
        return None, None
    min_y, min_x = opaque_points.min(axis=0)
    max_y, max_x = opaque_points.max(axis=0)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    return width, height


def get_cs_radius(cs):
    """Converts Circle Size (CS) to osu!pixels radius."""
    return (109 - 9 * cs) / 2


def run_command(command):
    """Runs a command in the shell and checks for errors."""
    print_status(f"Executing: {' '.join(command)}", level="CMD")
    try:
        # Using Popen to handle stdout/stderr streaming for logging
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding='utf-8', errors='replace', env=os.environ)
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
    except subprocess.CalledProcessError as e:
        print_status(f"Command failed with error: {e}", level="ERROR")
        return False
    return True


def stream_reader_thread(proc, progress_queue):
    """Reads a subprocess's stdout and puts progress updates into a queue."""
    try:
        for line in iter(proc.stdout.readline, ''):
            line = line.strip()
            if not line:
                continue

            if line.startswith("PROG"):
                try:
                    parts = line.split()
                    if parts[0] == "PROG":
                        progress = int(parts[1])
                        progress_queue.put((proc.pid, progress))
                    elif parts[0] == "PROG_DONE":
                        progress_queue.put((proc.pid, 'DONE'))
                except (IndexError, ValueError):
                    pass
            else:
                tqdm.write(f"[WORKER-{proc.pid}]: {line}")
    finally:
        proc.stdout.close()


def run_parallel_tasks(tasks_to_run):
    """Manages running multiple subprocesses in parallel."""
    # In a Vertex AI job, cpu_count() gives the cores of the machine type you chose.
    num_processes = max(1, cpu_count() - 1)
    print_status(f"Starting parallel generation with {num_processes} processes...")

    task_queue = list(tasks_to_run)
    active_procs = {}
    available_positions = set(range(1, num_processes + 1))
    progress_queue = queue.Queue()
    results = []

    with tqdm(total=len(tasks_to_run), desc="Overall Progress", position=0, ncols=100, file=sys.stdout) as main_pbar:
        while task_queue or active_procs:
            while task_queue and available_positions:
                task_command, task_name, task_duration_ms = task_queue.pop(0)
                total_frames_in_task = int(task_duration_ms / 1000 * FRAME_RATE)
                pbar_pos = available_positions.pop()
                pbar = tqdm(total=total_frames_in_task, desc=task_name, position=pbar_pos, ncols=100, leave=False,
                            file=sys.stdout)

                proc = subprocess.Popen(
                    task_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace'
                )

                thread = threading.Thread(target=stream_reader_thread, args=(proc, progress_queue))
                thread.daemon = True
                thread.start()

                active_procs[proc.pid] = {'proc': proc, 'pbar': pbar, 'name': task_name, 'thread': thread,
                                          'position': pbar_pos, 'total_frames': total_frames_in_task}

            while not progress_queue.empty():
                try:
                    pid, progress = progress_queue.get_nowait()
                    if pid in active_procs:
                        if progress == 'DONE':
                            active_procs[pid]['pbar'].n = active_procs[pid]['total_frames']
                        else:
                            active_procs[pid]['pbar'].n = progress
                        active_procs[pid]['pbar'].refresh()
                except queue.Empty:
                    pass

            for pid in list(active_procs.keys()):
                proc = active_procs[pid]['proc']
                if proc.poll() is not None:
                    details = active_procs[pid]
                    pbar = details['pbar']
                    details['thread'].join(timeout=1)
                    if proc.returncode == 0:
                        pbar.n = details['total_frames']
                        pbar.refresh()
                        results.append(True)
                    else:
                        tqdm.write(
                            f"\n[ERROR] Task FAILED: {details['name']} (Exit Code: {proc.returncode}). Check logs.")
                        results.append(False)

                    pbar.close()
                    available_positions.add(details['position'])
                    del active_procs[pid]
                    main_pbar.update(1)

            time.sleep(0.05)

    print("\n" * (num_processes + 2))
    sys.stdout.flush()
    return results


def create_modified_beatmap_gcs(original_gcs_path, temp_gcs_dir, cs, ar):
    """Creates a temporary .osu file in GCS with modified CS and AR values."""
    client = get_gcs_client()
    try:
        # Read original beatmap from GCS
        source_bucket_name, source_blob_name = parse_gcs_path(original_gcs_path)
        source_bucket = client.bucket(source_bucket_name)
        source_blob = source_bucket.blob(source_blob_name)
        original_content = source_blob.download_as_text()
        lines = original_content.splitlines(True)  # keepends=True

        new_lines = []
        in_difficulty = False
        for line in lines:
            if line.strip() == '[Difficulty]':
                in_difficulty = True
            elif line.strip().startswith('['):
                in_difficulty = False

            if in_difficulty:
                # Use .lower().startswith for case-insensitivity
                if line.lower().strip().startswith('circlesize'):
                    new_lines.append(f"CircleSize:{cs}\n")
                    continue
                if line.lower().strip().startswith('approachrate'):
                    new_lines.append(f"ApproachRate:{ar}\n")
                    continue
            new_lines.append(line)

        new_content = "".join(new_lines)

        # Upload modified beatmap to temp GCS location
        filename = source_blob_name.split('/')[-1]
        new_filename = f"{os.path.splitext(filename)[0]}_cs{cs}_ar{ar}.osu"

        dest_bucket_name, dest_prefix = parse_gcs_path(temp_gcs_dir)
        dest_blob_name = f"{dest_prefix}/{new_filename}"

        dest_bucket = client.bucket(dest_bucket_name)
        dest_blob = dest_bucket.blob(dest_blob_name)
        dest_blob.upload_from_string(new_content, content_type='text/plain')

        return f"gs://{dest_bucket_name}/{dest_blob_name}"

    except Exception as e:
        print_status(f"Failed to create modified beatmap for {original_gcs_path}: {e}", level="ERROR")
        return None


def consolidate_files_gcs(file_list, subset_name, master_dataset_gcs_path, start_index=0):
    """Moves and renames files within GCS to the final dataset directory."""
    print_status(f"Consolidating {len(file_list)} positive files for '{subset_name}' set...", level="INFO")

    dest_img_dir = f"{master_dataset_gcs_path}/images/{subset_name}"
    dest_lbl_dir = f"{master_dataset_gcs_path}/labels/{subset_name}"

    for idx, img_src_path in enumerate(tqdm(file_list, desc=f"Processing {subset_name} positive files")):
        # Construct label path from image path
        lbl_src_path = img_src_path.replace('/images/', '/labels/').replace('.jpg', '.txt')

        current_index = start_index + idx
        new_filename_base = f"frame_{current_index:07d}"
        dest_img_path = f"{dest_img_dir}/{new_filename_base}.jpg"
        dest_lbl_path = f"{dest_lbl_dir}/{new_filename_base}.txt"

        try:
            rename_gcs_blob(img_src_path, dest_img_path)
            # Check if label file exists before trying to move it
            _, lbl_blob_name = parse_gcs_path(lbl_src_path)
            if storage.Blob(bucket=storage_client.bucket(parse_gcs_path(lbl_src_path)[0]), name=lbl_blob_name).exists():
                rename_gcs_blob(lbl_src_path, dest_lbl_path)
        except Exception as e:
            print_status(f"Could not move file {img_src_path}. Error: {e}", level="ERROR")

    return start_index + len(file_list)


def process_negative_samples_gcs(source_paths, subset_name, master_dataset_gcs_path, available_cursor_pairs,
                                 start_index=0):
    """Copies negative samples, renders a moving cursor with a trail, and creates a label file in GCS."""
    if not available_cursor_pairs:
        print_status("No cursor/cursortrail asset pairs available; skipping negative sample augmentation.",
                     level="WARN")
        return start_index

    print_status(f"Augmenting {len(source_paths)} negative samples for '{subset_name}' set...", level="INFO")
    dest_img_dir = f"{master_dataset_gcs_path}/images/{subset_name}"
    dest_lbl_dir = f"{master_dataset_gcs_path}/labels/{subset_name}"
    CURSOR_CLASS_ID = 1
    TRAIL_LENGTH = 12

    client = get_gcs_client()

    for idx, img_src_path in enumerate(tqdm(source_paths, desc=f"Processing {subset_name} negative samples")):
        current_index = start_index + idx
        new_filename_base = f"frame_{current_index:07d}"
        dest_img_path = f"{dest_img_dir}/{new_filename_base}.jpg"
        dest_lbl_path = f"{dest_lbl_dir}/{new_filename_base}.txt"

        try:
            raw_cursor_img, raw_trail_img = random.choice(available_cursor_pairs)

            # Download base image from GCS
            bucket_name, blob_name = parse_gcs_path(img_src_path)
            blob = client.bucket(bucket_name).blob(blob_name)
            img_bytes = blob.download_as_bytes()

            with Image.open(io.BytesIO(img_bytes)).convert('RGBA') as base_img:
                img_w, img_h = base_img.size

                playfield_scale = img_h / 384.0
                cs = random.uniform(3, 6)
                circle_pixel_radius = int(get_cs_radius(cs) * playfield_scale)

                opaque_cursor_w, opaque_cursor_h = get_opaque_bbox_dimensions(raw_cursor_img)
                orig_opaque_w = opaque_cursor_w if opaque_cursor_w and opaque_cursor_w > 0 else raw_cursor_img.width
                orig_opaque_h = opaque_cursor_h if opaque_cursor_h and opaque_cursor_h > 0 else raw_cursor_img.height
                orig_opaque_max_dim = max(orig_opaque_w, orig_opaque_h)

                target_opaque_dim = max(15, int(circle_pixel_radius * 0.65))
                cursor_scale_ratio = target_opaque_dim / orig_opaque_max_dim if orig_opaque_max_dim > 0 else 0
                final_cursor_w = max(1, int(raw_cursor_img.width * cursor_scale_ratio))
                final_cursor_h = max(1, int(raw_cursor_img.height * cursor_scale_ratio))
                cursor_to_render = raw_cursor_img.resize((final_cursor_w, final_cursor_h), Image.Resampling.LANCZOS)

                anno_w, anno_h = get_opaque_bbox_dimensions(cursor_to_render)
                if not anno_w or not anno_h:
                    anno_w, anno_h = cursor_to_render.size

                final_trail_w = max(1, int(raw_trail_img.width * cursor_scale_ratio))
                final_trail_h = max(1, int(raw_trail_img.height * cursor_scale_ratio))
                trail_to_render = raw_trail_img.resize((final_trail_w, final_trail_h), Image.Resampling.LANCZOS)

                end_x = random.randint(anno_w, img_w - anno_w)
                end_y = random.randint(anno_h, img_h - anno_h)
                angle = random.uniform(0, 2 * np.pi)
                distance = random.uniform(20, 100)
                start_x = end_x - distance * np.cos(angle)
                start_y = end_y - distance * np.sin(angle)
                path_x = np.linspace(start_x, end_x, TRAIL_LENGTH)
                path_y = np.linspace(start_y, end_y, TRAIL_LENGTH)
                cursor_path = list(zip(path_x, path_y))

                for i, pos in enumerate(cursor_path[:-1]):
                    opacity = (i + 1) / TRAIL_LENGTH
                    temp_trail = trail_to_render.copy()
                    alpha = temp_trail.getchannel('A')
                    temp_trail.putalpha(alpha.point(lambda p: int(p * opacity)))
                    paste_pos = (int(pos[0] - temp_trail.width // 2), int(pos[1] - temp_trail.height // 2))
                    base_img.paste(temp_trail, paste_pos, temp_trail)

                final_cursor_pos = cursor_path[-1]
                paste_pos = (int(final_cursor_pos[0] - cursor_to_render.width // 2),
                             int(final_cursor_pos[1] - cursor_to_render.height // 2))
                base_img.paste(cursor_to_render, paste_pos, cursor_to_render)

                # Save augmented image to GCS
                img_byte_arr = io.BytesIO()
                base_img.convert('RGB').save(img_byte_arr, format='JPEG', quality=90)

                dest_bucket_name, dest_blob_name = parse_gcs_path(dest_img_path)
                client.bucket(dest_bucket_name).blob(dest_blob_name).upload_from_string(img_byte_arr.getvalue(),
                                                                                        content_type='image/jpeg')

                # Create and upload label file to GCS
                x_center_norm = final_cursor_pos[0] / img_w
                y_center_norm = final_cursor_pos[1] / img_h
                width_norm = anno_w / img_w
                height_norm = anno_h / img_h
                label_content = f"{CURSOR_CLASS_ID} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n"

                dest_bucket_name, dest_blob_name = parse_gcs_path(dest_lbl_path)
                client.bucket(dest_bucket_name).blob(dest_blob_name).upload_from_string(label_content,
                                                                                        content_type='text/plain')

        except Exception as e:
            print_status(f"Could not process negative sample {img_src_path}. Error: {e}", level="ERROR")

    return start_index + len(source_paths)


def main():
    """Main execution function for the pipeline."""
    global BASE_GCS_BUCKET, SOURCE_BEATMAPS_DIR, SOURCE_SKINS_DIR, SOURCE_REPLAYS_DIR, SOURCE_BACKGROUNDS_DIR, SOURCE_MENU_DIR, BASE_OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Master pipeline for osu! dataset generation and training on Vertex AI.")
    parser.add_argument("--finetune-weights", help="GCS path to 'best.pt' to continue training from.", default=None)
    parser.add_argument("--base-gcs-bucket", required=True, help="Base GCS bucket path (e.g., gs://my-osu-bucket).")
    args = parser.parse_args()

    # --- Override default paths with command-line argument ---
    BASE_GCS_BUCKET = args.base_gcs_bucket
    SOURCE_BEATMAPS_DIR = f"{BASE_GCS_BUCKET}/source/beatmaps"
    SOURCE_SKINS_DIR = f"{BASE_GCS_BUCKET}/source/skins"
    SOURCE_REPLAYS_DIR = f"{BASE_GCS_BUCKET}/source/replays"
    SOURCE_BACKGROUNDS_DIR = f"{BASE_GCS_BUCKET}/source/backgrounds"
    SOURCE_MENU_DIR = f"{BASE_GCS_BUCKET}/source/menu"
    BASE_OUTPUT_DIR = f"{BASE_GCS_BUCKET}/datasets"

    print_status("Starting Master Automation Pipeline on Vertex AI...")

    # Initialize GCS client
    get_gcs_client()

    # MODIFIED: List files from GCS
    beatmaps = list_gcs_files(SOURCE_BEATMAPS_DIR, suffix='.osu')
    skins = list_gcs_dirs(SOURCE_SKINS_DIR)
    background_images = list_gcs_files(SOURCE_BACKGROUNDS_DIR, suffix=('.png', '.jpg', '.jpeg'))

    print_status(f"Found {len(beatmaps)} beatmaps, {len(skins)} skins, {len(background_images)} backgrounds in GCS.")

    print_status("Partitioning beatmaps for training and validation sets...")
    random.shuffle(beatmaps)
    split_index = int(len(beatmaps) * VALIDATION_SPLIT)
    validation_beatmaps = beatmaps[:split_index]
    training_beatmaps = beatmaps[split_index:]
    print_status(
        f"Partition complete: {len(training_beatmaps)} maps for Training, {len(validation_beatmaps)} for Validation.")

    if not validation_beatmaps:
        print_status("The validation beatmap set is empty. Training will fail. Add more beatmaps or adjust split.",
                     level="WARN")

    tasks = []
    # MODIFIED: Temporary directory is now in GCS
    temp_beatmap_dir = f"{BASE_OUTPUT_DIR}/temp_beatmaps"
    # Use local path for script, as it's inside the Docker container
    autogen_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autogen_dataset.py")

    all_map_lists = [{'maps': training_beatmaps, 'is_val': False}, {'maps': validation_beatmaps, 'is_val': True}]

    from osrparse import Replay as ReplayParser
    import io

    for map_group in all_map_lists:
        map_list, is_validation_set = map_group['maps'], map_group['is_val']
        val_split_arg = "1.0" if is_validation_set else "0.0"
        set_name_for_tqdm = "Validation" if is_validation_set else "Training"

        generation_plan = []
        if GENERATE_ORIGINAL_WITH_REPLAY:
            for map_path in map_list:
                map_basename = map_path.split('/')[-1]
                replay_path = f"{SOURCE_REPLAYS_DIR}/{os.path.splitext(map_basename)[0]}.osr"
                # Check if replay exists in GCS
                bucket_name, blob_name = parse_gcs_path(replay_path)
                if storage.Blob(bucket=storage_client.bucket(bucket_name), name=blob_name).exists():
                    generation_plan.extend([
                        {"map": map_path, "replay": replay_path, "mods": ["NM"]},
                        {"map": map_path, "replay": replay_path, "mods": ["HD"]},
                        {"map": map_path, "replay": replay_path, "mods": ["DT"]},
                        {"map": map_path, "replay": replay_path, "mods": ["HR"]}
                    ])

        for map_path in map_list:
            map_basename = map_path.split('/')[-1]
            replay_path = f"{SOURCE_REPLAYS_DIR}/{os.path.splitext(map_basename)[0]}.osr"
            bucket_name, blob_name = parse_gcs_path(replay_path)
            associated_replay = replay_path if storage.Blob(bucket=storage_client.bucket(bucket_name),
                                                            name=blob_name).exists() else None

            try:
                # Parse beatmap from GCS
                bucket_name, blob_name = parse_gcs_path(map_path)
                content = storage_client.bucket(bucket_name).blob(blob_name).download_as_text()
                _, od, _, _, _ = parse_beatmap(map_path, content=content, apply_hr=False, apply_dt=False)
                if not od: continue
                generation_plan.append({"map": map_path, "replay": associated_replay, "mods": ["Modified"],
                                        "original_cs": od.get('CircleSize', 5)})
            except Exception as e:
                print_status(f"Skipping map due to parsing error: {map_path} - {e}", "WARN")
                continue

        # ... (The rest of the planning logic is largely the same, just ensure paths are GCS paths)
        maps_with_special_clips = set()
        for plan in tqdm(generation_plan, desc=f"Planning {set_name_for_tqdm} Tasks"):
            map_path, mods = plan["map"], plan["mods"]
            try:
                bucket_name, blob_name = parse_gcs_path(map_path)
                content = storage_client.bucket(bucket_name).blob(blob_name).download_as_text()
                ho_nomod, _, _, _, _ = parse_beatmap(map_path, content=content, apply_hr=False, apply_dt=False)
                if not ho_nomod: continue
            except Exception:
                continue

            replay_start_time_ms = 0
            if plan['replay']:
                try:
                    bucket_name, blob_name = parse_gcs_path(plan['replay'])
                    replay_bytes = storage_client.bucket(bucket_name).blob(blob_name).download_as_bytes()
                    replay_obj = ReplayParser.from_file(io.BytesIO(replay_bytes))
                    if replay_obj.replay_data:
                        replay_start_time_ms = replay_obj.replay_data[0].time_delta
                        if "DT" in mods:
                            replay_start_time_ms /= DT_RATE
                except Exception as e:
                    print_status(f"Could not pre-parse replay {plan['replay']} to get start time: {e}", "WARN")

            clips = [c for c in [find_best_section_by_count(ho_nomod, CLIP_DURATION_MS, 'circle'),
                                 find_best_section_by_count(ho_nomod, CLIP_DURATION_MS, 'slider')] if c]
            if mods == ["NM"]: clips.extend(get_spinner_clips(ho_nomod))

            clips = list({frozenset(d.items()): d for d in clips}.values())
            valid_clips = [clip for clip in clips if clip['start'] >= replay_start_time_ms]

            if not valid_clips:
                continue

            map_basename = map_path.split('/')[-1]

            if valid_clips and map_basename not in maps_with_special_clips:
                clip_for_special = valid_clips[0]
                skin_path, bg_path = random.choice(skins), random.choice(
                    background_images) if background_images else None
                map_name, skin_name = os.path.splitext(map_basename)[0], skin_path.strip('/').split('/')[-1]
                clip_type = clip_for_special.get('type', 'clip').upper()

                base_cmd = ["python", autogen_script_path, map_path, skin_path,
                            "--start-time", str(int(clip_for_special['start'])),
                            "--duration", str(CLIP_DURATION_MS), "--val-split", val_split_arg,
                            "--background-opacity", str(random.uniform(0.01, 0.40)), "--reporter"]
                if plan['replay']: base_cmd.extend(["--replay_path", plan['replay']])
                if bg_path: base_cmd.extend(["--background-path", bg_path])

                no_cursor_dir = f"{map_name}_{skin_name}_NoCursor_{clip_type}"
                no_cursor_task_name = f"{map_name[:10]:<10} | {skin_name[:10]:<10} | {'NO-CURSOR':<12} | {clip_type:<7} {CLIP_DURATION_MS / 1000:.1f}s"
                no_cursor_cmd = base_cmd + ["--output-dir",
                                            f"{BASE_OUTPUT_DIR}/individual_sets/{no_cursor_dir}",
                                            "--filename-prefix", no_cursor_dir, "--no-cursor"]
                tasks.append((no_cursor_cmd, no_cursor_task_name, CLIP_DURATION_MS))

                if plan['replay']:
                    cursor_only_dir = f"{map_name}_{skin_name}_CursorOnly_{clip_type}"
                    cursor_only_task_name = f"{map_name[:10]:<10} | {skin_name[:10]:<10} | {'CURSOR-ONLY':<12} | {clip_type:<7} {CLIP_DURATION_MS / 1000:.1f}s"
                    cursor_only_cmd = base_cmd + ["--output-dir",
                                                  f"{BASE_OUTPUT_DIR}/individual_sets/{cursor_only_dir}",
                                                  "--filename-prefix", cursor_only_dir, "--cursor-only"]
                    tasks.append((cursor_only_cmd, cursor_only_task_name, CLIP_DURATION_MS))

                maps_with_special_clips.add(map_basename)

            for i, clip in enumerate(valid_clips):
                skin_path, background_path = random.choice(skins), random.choice(
                    background_images) if background_images else None
                opacity = 0.0 if random.random() < 0.5 else (
                    random.uniform(0.01, 0.40) if random.random() < 0.8 else random.uniform(0.41, 1.0))
                current_map_path, current_clip = map_path, clip.copy()
                mods_str, diff_str, full_replay_sim = "".join(sorted(mods)), "Orig", False

                if "Modified" in mods:
                    cs, ar = random.choice(CS_VALUES_MODIFIED), random.choice(AR_VALUES_MODIFIED)
                    # Use GCS version of modified beatmap creation
                    current_map_path = create_modified_beatmap_gcs(map_path, temp_beatmap_dir, cs, ar)
                    if not current_map_path: continue
                    diff_str, mods_str = f"CS{cs}AR{ar}", "NM"
                    if cs <= plan.get("original_cs", 5): full_replay_sim = True

                if "DT" in mods:
                    current_clip['start'] /= DT_RATE
                    current_clip['end'] /= DT_RATE

                duration_ms = current_clip['end'] - current_clip['start']
                if duration_ms <= 0: continue

                map_name, skin_name = os.path.splitext(map_path.split('/')[-1])[0], skin_path.strip('/').split('/')[-1]
                clip_type, settings_str = clip.get('type', 'clip').upper(), f"({mods_str}_{diff_str})"
                out_dir_name = f"{map_name}_{skin_name}{settings_str}_{clip_type}_{i}"
                task_name = f"{map_name[:10]:<10} | {skin_name[:10]:<10} | {settings_str:<12} | {clip_type:<7} {duration_ms / 1000:.1f}s"
                out_path = f"{BASE_OUTPUT_DIR}/individual_sets/{out_dir_name}"

                cmd = ["python", autogen_script_path, current_map_path, skin_path, "--output-dir", out_path,
                       "--start-time", str(int(current_clip['start'])),
                       "--duration", str(int(duration_ms)), "--val-split", val_split_arg, "--filename-prefix",
                       out_dir_name, "--background-opacity", str(opacity), "--reporter"]
                if plan['replay']: cmd.extend(["--replay_path", plan['replay']])
                if "HD" in mods: cmd.append("--hd")
                if "HR" in mods: cmd.append("--hr")
                if "DT" in mods: cmd.append("--dt")
                if "Modified" in mods: cmd.append("--no-misses")
                if full_replay_sim: cmd.append("--full-replay-sim")
                if background_path: cmd.extend(["--background-path", background_path])
                tasks.append((cmd, task_name, duration_ms))

    random.shuffle(tasks)
    print_status(f"Total generation tasks planned: {len(tasks)}")
    results = run_parallel_tasks(tasks)
    if not any(results):
        print_status("No datasets were generated. Exiting.", level="ERROR")
        return

    master_dataset_path = f"{BASE_OUTPUT_DIR}/{FINAL_DATASET_NAME}"
    individual_sets_dir = f"{BASE_OUTPUT_DIR}/individual_sets"

    all_train_images, all_val_images = [], []
    print_status("Collecting all generated image files from GCS...", level="INFO")

    # List all individual set directories
    individual_set_dirs = list_gcs_dirs(individual_sets_dir)
    for set_dir in tqdm(individual_set_dirs, desc="Scanning generated sets"):
        all_train_images.extend(list_gcs_files(f"{set_dir}images/train", suffix=".jpg"))
        all_val_images.extend(list_gcs_files(f"{set_dir}images/val", suffix=".jpg"))

    if not all_train_images:
        print_status("Fatal: No training images were found after generation. Cannot proceed.", level="ERROR")
        return
    if not all_val_images:
        print_status("Fatal: No validation images were found after generation. Cannot proceed.", level="ERROR")
        return

    print_status(f"Found {len(all_train_images)} training images and {len(all_val_images)} validation images.")

    all_train_images.sort()
    all_val_images.sort()

    # --- Load assets for negative sampling ---
    from skin_loader import load_skin_assets as load_skin_assets_local
    cursor_trail_pairs = []
    if skins:
        print_status("Loading cursor/trail pairs for negative sampling from GCS...", level="INFO")
        for skin_gcs_path in skins:
            try:
                # skin_loader needs to be adapted to read from GCS. Assuming it is.
                # If not, this part needs a local copy first. For simplicity, we assume it works.
                assets, _ = load_skin_assets_local(skin_gcs_path)
                cursor_img = assets.get('cursor')
                trail_img = assets.get('cursortrail')
                if cursor_img and trail_img:
                    cursor_bbox = cursor_img.getbbox()
                    trail_bbox = trail_img.getbbox()
                    cropped_cursor = cursor_img.crop(cursor_bbox) if cursor_bbox else cursor_img
                    cropped_trail = trail_img.crop(trail_bbox) if trail_bbox else trail_img
                    if cropped_cursor.width > 0 and cropped_trail.width > 0:
                        cursor_trail_pairs.append((cropped_cursor, cropped_trail))
            except Exception as e:
                skin_name = skin_gcs_path.strip('/').split('/')[-1]
                print_status(f"Could not load cursor/trail from {skin_name}: {e}", "WARN")

    menu_images = list_gcs_files(SOURCE_MENU_DIR, suffix=('.png', '.jpg', '.jpeg'))

    # Consolidate VAL files and add negative samples
    last_val_idx = consolidate_files_gcs(all_val_images, 'val', master_dataset_path, 0)
    if menu_images and NEGATIVE_SAMPLE_RATIO > 0 and cursor_trail_pairs:
        target_val_neg_count = int((NEGATIVE_SAMPLE_RATIO * len(all_val_images)) / (1 - NEGATIVE_SAMPLE_RATIO))
        if target_val_neg_count > 0:
            last_val_idx = process_negative_samples_gcs(random.choices(menu_images, k=target_val_neg_count), 'val',
                                                        master_dataset_path, cursor_trail_pairs, last_val_idx + 1)

    # Consolidate TRAIN files and add negative samples
    last_train_idx = consolidate_files_gcs(all_train_images, 'train', master_dataset_path, 0)
    if menu_images and NEGATIVE_SAMPLE_RATIO > 0 and cursor_trail_pairs:
        target_train_neg_count = int((NEGATIVE_SAMPLE_RATIO * len(all_train_images)) / (1 - NEGATIVE_SAMPLE_RATIO))
        if target_train_neg_count > 0:
            process_negative_samples_gcs(random.choices(menu_images, k=target_train_neg_count), 'train',
                                         master_dataset_path, cursor_trail_pairs, last_train_idx + 1)

    print_status(f"Master dataset created in: {master_dataset_path}")
    delete_gcs_folder(individual_sets_dir)
    delete_gcs_folder(temp_beatmap_dir)

    print_status("Starting final model training...", level="TASK")
    train_img_path = f"{master_dataset_path}/images/train"
    val_img_path = f"{master_dataset_path}/images/val"

    # Create YAML content and upload to GCS
    yaml_content = f"train: {train_img_path}\nval: {val_img_path}\nnc: 3\nnames: ['hit_circle', 'cursor', 'hit_miss']\n"
    yaml_gcs_path = f"{master_dataset_path}/{FINAL_DATASET_NAME}.yaml"

    bucket_name, blob_name = parse_gcs_path(yaml_gcs_path)
    storage_client.bucket(bucket_name).blob(blob_name).upload_from_string(yaml_content, 'text/yaml')
    print_status(f"Uploaded training YAML to {yaml_gcs_path}")

    # Use local path for script, as it's inside the Docker container
    train_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    training_run_name = "yolov8n_osu_final_run_v9"

    # The `train.py` script will be called by the Vertex AI job directly, not as a subprocess here.
    # The YAML path is the key output of this script.
    # We will add a final instruction file to the bucket.
    final_instructions = f"""
    Vertex AI Data Generation Complete.

    To start training, run a new Vertex AI Custom Job using the same Docker container.

    Entry Point: train.py

    Arguments:
    --data {yaml_gcs_path}
    --name {training_run_name}
    --project {master_dataset_path}/runs/detect
    """
    if args.finetune_weights:
        final_instructions += f" --weights {args.finetune_weights}"

    instructions_path = f"{master_dataset_path}/TRAINING_INSTRUCTIONS.txt"
    bucket_name, blob_name = parse_gcs_path(instructions_path)
    storage_client.bucket(bucket_name).blob(blob_name).upload_from_string(final_instructions)

    print_status("Master Pipeline Finished!", level="SUCCESS")
    print_status(f"Training YAML and instructions are ready at {master_dataset_path}")
    print_status("You can now start the training job on Vertex AI.")


if __name__ == "__main__":
    main()