import cv2
import numpy as np
import json
import os
import time
import multiprocessing
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from collections import deque

# Local imports
from config_manager import ConfigManager
from recognition import OsuRecognitionSystem
import video_processing
import visualization  # BUG FIX: Added missing import

# --- Constants for tracking ---
TRACKING_MAX_DISTANCE = 50
TRACKING_UNSEEN_FRAMES_TOLERANCE = 5


class AnalysisEngine:
    """
    Handles the core analysis of the gameplay video, including object detection,
    lifecycle tracking, and data aggregation.
    """

    def __init__(self, video_path: str, ocr_preset_name: Optional[str], config_manager: ConfigManager):
        self.video_path = video_path
        self.ocr_preset_name = ocr_preset_name
        self.config_manager = config_manager

        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap.release()

        self.output_dir = Path("src/debug_output")
        self.recognition_system = OsuRecognitionSystem(debug_mode=True)

        # Load calibration and OCR settings
        self.calibration_data = self._load_calibration_data()
        self.ui_regions = self.calibration_data.get('ui_regions', {})
        self.hit_circle_params = self.calibration_data.get('hit_circle_params', {})
        self._load_ocr_settings()

        # Analysis state
        self.active_circles = {}
        self.completed_circles = []
        self.next_circle_id = 0

        self.analysis_data = {
            "video_info": {"path": video_path, "width": self.width, "height": self.height, "fps": self.fps,
                           "total_frames": self.total_frames},
            "ui_regions": {k: v for k, v in self.ui_regions.items() if v is not None},
            "ocr_preset_used": self.ocr_preset_name,
            "data_points": [],
            "hit_circles": [],
        }

    def _load_calibration_data(self) -> Dict:
        config_path = self.output_dir / "calibration_data.json"
        if config_path.exists():
            with open(config_path, 'r') as f: return json.load(f)
        return {}

    def _load_ocr_settings(self):
        """Loads OCR settings based on the specified preset or defaults."""
        self.current_ocr_settings: Dict[str, Any] = {
            'combo': self.config_manager.get_default_combo_settings(),
            'accuracy': self.config_manager.get_default_accuracy_settings(),
        }
        if self.ocr_preset_name:
            preset_data = self.config_manager.get_preset(self.ocr_preset_name)
            if preset_data:
                for region, settings in preset_data.items():
                    if region in self.current_ocr_settings:
                        self.current_ocr_settings[region].update(settings)
                print(f"Loaded and applied OCR preset: '{self.ocr_preset_name}'")
            else:
                print(f"Warning: OCR preset '{self.ocr_preset_name}' not found.")

    def run_full_analysis(self):
        """Orchestrates the entire analysis process."""
        print(f"\nStarting video analysis for: {self.video_path}")

        # 1. Parallel Detection
        num_workers = max(1, os.cpu_count() - 1)
        print(f"Using {num_workers} worker processes for detection.")
        frame_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        pool = multiprocessing.Pool(num_workers, video_processing.detection_worker,
                                    (self.video_path, self.ui_regions, self.hit_circle_params, frame_queue,
                                     result_queue))

        print("Loading frames into queue...")
        frames_to_process = range(0, self.total_frames, 2)  # Process every other frame
        for i in frames_to_process:
            frame_queue.put(i)
        for _ in range(num_workers):
            frame_queue.put(None)

        detection_results = []
        num_frames_to_process = len(frames_to_process)
        for i in range(num_frames_to_process):
            detection_results.append(result_queue.get())
            # Progress bar
            progress = (i + 1) / num_frames_to_process
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\rCollecting results: |{bar}| {progress:.1%} complete', end='', flush=True)

        print("\nDetection phase complete.")
        pool.close()
        pool.join()

        # 2. Sequential Processing and Tracking
        self.process_detection_results(detection_results)

        # 3. Save final data
        analysis_data_path = self.output_dir / "analysis_debug.json"
        with open(analysis_data_path, 'w') as f:
            json.dump(self.analysis_data, f, indent=4)
        print(f"Analysis data saved to {analysis_data_path}")

        # 4. Generate visualization
        vis = visualization.Visualization(analysis_data_path)
        vis.create_data_plot()

    def process_detection_results(self, detection_results: List[Dict]):
        """
        Processes the raw detection results to perform lifecycle tracking and final analysis.
        """
        print("Starting sequential analysis and lifecycle tracking...")
        detection_results.sort(key=lambda x: x['frame'])

        last_combo_hash = None
        last_acc_hash = None

        for result in detection_results:
            frame_num = result['frame']
            timestamp = frame_num / self.fps

            # Perform Circle Lifecycle Tracking
            if self.hit_circle_params:
                detected_circles = [tuple(c) for c in result.get('circles', [])]
                self._update_circle_tracker(detected_circles, timestamp)

            # Perform Selective OCR
            current_data_point = {"frame": frame_num, "timestamp": timestamp, "combo": None, "accuracy": None}
            # ... (OCR logic from original analyzer's process_video_full) ...
            self.analysis_data["data_points"].append(current_data_point)

        # Finalize any remaining active circles
        for circle_id in list(self.active_circles.keys()):
            vanished_circle = self.active_circles.pop(circle_id)
            self.completed_circles.append({
                "id": vanished_circle['id'], "x": int(vanished_circle['pos'][0]), "y": int(vanished_circle['pos'][1]),
                "start_ts": vanished_circle['start_ts'], "end_ts": vanished_circle['last_seen_ts']
            })

        self.analysis_data['hit_circles'] = self.completed_circles
        print(f"Found {len(self.analysis_data['hit_circles'])} hit circles.")

    def _update_circle_tracker(self, detected_circles: List[Tuple[int, int, int]], timestamp: float):
        """Updates the state of tracked circles based on newly detected circles."""
        # This is the core tracking logic moved from DebugOsuAnalyzer._update_circle_tracker
        detected_centers = np.array([[c[0], c[1]] for c in detected_circles]) if detected_circles else np.empty((0, 2))
        unmatched_detections = list(range(len(detected_centers)))

        if self.active_circles and detected_circles:
            active_ids = list(self.active_circles.keys())
            active_centers = np.array([self.active_circles[id]['pos'] for id in active_ids])
            dist_matrix = np.linalg.norm(active_centers[:, np.newaxis, :] - detected_centers[np.newaxis, :, :], axis=2)

            # Greedy matching
            while np.min(dist_matrix) < TRACKING_MAX_DISTANCE:
                min_val = np.min(dist_matrix)
                if min_val >= TRACKING_MAX_DISTANCE: break
                row_idx, col_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                active_id = active_ids[row_idx]
                detection_idx = col_idx

                new_pos = detected_centers[detection_idx]
                self.active_circles[active_id]['pos'] = [int(new_pos[0]), int(new_pos[1])]
                self.active_circles[active_id]['last_seen_ts'] = timestamp
                self.active_circles[active_id]['unseen_frames'] = 0

                if detection_idx in unmatched_detections:
                    unmatched_detections.remove(detection_idx)

                dist_matrix[row_idx, :] = np.inf
                dist_matrix[:, col_idx] = np.inf

        for circle_id in list(self.active_circles.keys()):
            if self.active_circles[circle_id]['last_seen_ts'] < timestamp:
                self.active_circles[circle_id]['unseen_frames'] += 1
            if self.active_circles[circle_id]['unseen_frames'] > TRACKING_UNSEEN_FRAMES_TOLERANCE:
                vanished_circle = self.active_circles.pop(circle_id)
                self.completed_circles.append({
                    "id": vanished_circle['id'], "x": int(vanished_circle['pos'][0]),
                    "y": int(vanished_circle['pos'][1]),
                    "start_ts": vanished_circle['start_ts'], "end_ts": vanished_circle['last_seen_ts']
                })

        for detection_idx in unmatched_detections:
            new_pos = detected_centers[detection_idx]
            self.active_circles[self.next_circle_id] = {
                'id': self.next_circle_id, 'pos': [int(new_pos[0]), int(new_pos[1])],
                'start_ts': timestamp, 'last_seen_ts': timestamp, 'unseen_frames': 0
            }
            self.next_circle_id += 1
