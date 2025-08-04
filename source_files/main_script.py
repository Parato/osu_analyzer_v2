# main_script.py
#
# Usage
# cd source_files
# python main_script.py video/test5.mp4 --output-video out.mp4
#
# MODIFIED: This script has been refactored to be a pure detection and tracking tool.
# It no longer requires a beatmap file and focuses on identifying objects in the video.

import argparse
import cv2
from tqdm import tqdm
import os
import copy

# --- FIX for OMP Error #15 ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- Local Imports ---
from detector import CircleDetector
from tracker import ObjectTracker
from visualizer import Visualizer
import config
from utils import print_status, get_best_cursor_detections


def load_video(video_path):
    """Loads a video file using OpenCV."""
    print_status(f"Loading video from: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print_status(f"Error: Could not open video file at {video_path}", level="ERROR")
        return None, None

    video_info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }

    print_status("Video loaded successfully.")
    print_status(
        f"Properties: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.2f} FPS, {video_info['frame_count']} total frames.")

    return cap, video_info


def process_video(cap, video_info, args):
    """
    Main processing loop. Detects objects, tracks them, and creates a
    visualization video.
    """
    frame_number = 0
    total_frames = video_info['frame_count']

    # --- Pass 1: Detection and Tracking ---
    print_status("Starting Pass 1: Detection and Tracking...")
    detector = CircleDetector()
    tracker = ObjectTracker(max_age=args.tracker_age, dist_thresh=args.tracker_dist)
    all_detections_by_frame = {}
    all_tracks_by_frame = {}

    with tqdm(total=total_frames, unit="frame", desc="Pass 1/2: Detecting") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            all_detections_by_frame[frame_number] = detections
            tracker.update(detections, frame_number)
            all_tracks_by_frame[frame_number] = copy.deepcopy(tracker.active_tracks)

            frame_number += 1
            pbar.update(1)

    print_status("Finished detection pass.")

    # --- Get the single best cursor detection per frame for clean visualization ---
    cursor_detections = get_best_cursor_detections(all_detections_by_frame)

    # --- Pass 2: Visualization ---
    print_status("Starting Pass 2: Writing Output Video...")
    visualizer = Visualizer(args.output_video, video_info)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_number = 0

    with tqdm(total=total_frames, unit="frame", desc="Pass 2/2: Visualizing") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections_in_frame = all_detections_by_frame.get(frame_number, [])
            tracks_in_frame = all_tracks_by_frame.get(frame_number, {})

            # The visualizer now only needs the raw detection/track data.
            vis_frame = visualizer.draw_frame(
                frame.copy(),
                detections_in_frame,
                tracks_in_frame,
                cursor_detection=cursor_detections.get(frame_number)
            )
            visualizer.write(vis_frame)

            frame_number += 1
            pbar.update(1)

    visualizer.release()
    print_status("Finished processing video.")


def main():
    """The main function to orchestrate the entire process."""
    parser = argparse.ArgumentParser(
        description="Detects and tracks osu! gameplay objects from a video file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("--output-video", help="Path to save the output video with visualizations.", default=None)

    # --- Tracker-specific arguments from config file ---
    parser.add_argument("--tracker-age", type=int, default=config.MAX_TRACK_AGE,
                        help=f"Max frames to keep a track alive without new detection (default: {config.MAX_TRACK_AGE}).")
    parser.add_argument("--tracker-dist", type=int, default=config.TRACKING_DIST_THRESHOLD,
                        help=f"Max distance to associate a detection with a track (default: {config.TRACKING_DIST_THRESHOLD}).")

    args = parser.parse_args()

    # --- Load Video ---
    cap, video_info = load_video(args.video_path)
    if not cap:
        return

    # --- Beatmap data is no longer needed. Start processing directly. ---
    process_video(cap, video_info, args)

    # --- Cleanup ---
    cap.release()
    print_status("Script finished.")


if __name__ == "__main__":
    main()