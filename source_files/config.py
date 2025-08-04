# config.py
#
# Stores all configuration parameters for the detector and tracker.
# MODIFIED: YOLO_CONFIDENCE_THRESHOLD is now a dictionary to allow for
# class-specific confidence scores.
# MODIFIED: Lowered confidence thresholds and tracker parameters to better
# handle real-world video artifacts and motion blur.
# MODIFIED: Removed obsolete classes from confidence thresholds.

import os

# Get the absolute path of the directory where this config file is located.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# --- Deep Learning Detector Configuration ---
# Path to the trained YOLOv8 model file (.pt).
# This is now an absolute path, making it robust to where other scripts are run from.
YOLO_MODEL_PATH = os.path.join(SCRIPT_DIR, "runs/detect/test7/weights/best.pt")

# --- MODIFIED: Class-Specific Confidence Thresholds ---
# Lowered thresholds to be more tolerant of real-world video artifacts.
# The cursor, which suffers from motion blur, has the lowest threshold.
YOLO_CONFIDENCE_THRESHOLDS = {
    "default": 0.6,
    "cursor": 0.6,
    "hit_circle": 0.6,
}


# --- MODIFIED: Tracker Configuration ---
# Adjusted to be more strict, which is appropriate for a more responsive detector.
# If the detector fails to find an object for 10 consecutive frames, the track is dropped.
MAX_TRACK_AGE = 5
# The max pixel distance to associate a detection with an existing track.
TRACKING_DIST_THRESHOLD = 150