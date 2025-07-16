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
from ocr_calibrator import OCRCalibrator
from video_processing import detection_worker
import visualization
from difficulty_calculator import DifficultyCalculator

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

        self.output_dir = Path("src/saves/overlay")
        self.recognition_system = OsuRecognitionSystem(debug_mode=True)
        self.difficulty_calculator = DifficultyCalculator()

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
            "hit_objects": [],
            "star_rating": 0.0
        }
        self.ocr_calibrator = OCRCalibrator(video_path, config_manager)

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

        num_workers = max(1, os.cpu_count() - 1)
        print(f"Using {num_workers} worker processes for detection.")
        frame_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        worker_args = (self.video_path, self.ui_regions, self.hit_circle_params, frame_queue, result_queue)
        pool = multiprocessing.Pool(num_workers, detection_worker, worker_args)

        print("Loading frames into queue...")
        frames_to_process = range(0, self.total_frames, 2)
        for i in frames_to_process:
            frame_queue.put(i)
        for _ in range(num_workers):
            frame_queue.put(None)

        detection_results = []
        num_frames_to_process = len(frames_to_process)
        for i in range(num_frames_to_process):
            detection_results.append(result_queue.get())
            progress = (i + 1) / num_frames_to_process
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\rCollecting results: |{bar}| {progress:.1%} complete', end='', flush=True)

        print("\nDetection phase complete.")
        pool.close()
        pool.join()

        self.process_detection_results(detection_results)

        print("Calculating map difficulty...")
        circle_radius = self.hit_circle_params.get('maxRadius', 0.0)

        if circle_radius > 0:
            self.analysis_data['star_rating'] = self.difficulty_calculator.calculate_star_rating(
                self.analysis_data['hit_objects'],
                circle_radius=float(circle_radius)
            )
            print(f"Estimated Star Rating: {self.analysis_data['star_rating']:.2f} ★")
        else:
            self.analysis_data['star_rating'] = 0.0
            print("Could not calculate star rating: Hit circle radius is not calibrated.")

        analysis_output_dir = Path("src/saves/result")
        analysis_output_dir.mkdir(parents=True, exist_ok=True)
        analysis_data_path = analysis_output_dir / "analysis_debug.json"
        with open(analysis_data_path, 'w') as f:
            json.dump(self.analysis_data, f, indent=4)
        print(f"Analysis data saved to {analysis_data_path}")

        vis = visualization.Visualization(analysis_data_path)
        vis.create_data_plot()
        self.cap.release()

    def _normalize_circle_durations(self) -> List[Dict]:
        """
        Calculates the median duration of all detected circles and applies it
        to every circle for a standardized lifetime.
        """
        if not self.completed_circles:
            return []

        durations = [c['end_ts'] - c['start_ts'] for c in self.completed_circles]
        valid_durations = [d for d in durations if 0.2 < d < 2.0]

        if not valid_durations:
            median_duration = 0.8
            print("Warning: Could not determine median circle duration. Using default.")
        else:
            median_duration = np.median(valid_durations)
            print(f"Determined standard circle duration to be {median_duration:.3f} seconds.")

        normalized_hit_objects = []
        for circle in self.completed_circles:
            new_circle = circle.copy()
            new_circle['end_ts'] = circle['start_ts'] + median_duration
            normalized_hit_objects.append(new_circle)

        return normalized_hit_objects

    def process_detection_results(self, detection_results: List[Dict]):
        """
        Processes the raw detection results to perform lifecycle tracking and final analysis.
        """
        print("Starting sequential analysis and lifecycle tracking...")
        detection_results.sort(key=lambda x: x['frame'])

        last_combo_hash = None
        last_acc_hash = None
        last_combo_value = None
        last_acc_value = None

        total_raw_detections = 0
        for result in detection_results:
            frame_num = result['frame']
            timestamp = frame_num / self.fps

            detected_circles_in_frame = result.get('circles', [])
            total_raw_detections += len(detected_circles_in_frame)

            if self.hit_circle_params:
                detected_circles = [tuple(c) for c in detected_circles_in_frame]
                self._update_circle_tracker(detected_circles, timestamp)

            current_data_point = {"frame": frame_num, "timestamp": timestamp, "combo": None, "accuracy": None}

            combo_hash = result.get('combo_hash')
            if combo_hash is not None and combo_hash != last_combo_hash:
                combo_img = result.get('combo_img')
                if combo_img is not None:
                    processed_combo = self.ocr_calibrator.preprocess_for_ocr(combo_img, 'combo',
                                                                             self.current_ocr_settings['combo'])
                    combo_val = self.recognition_system.recognize_combo(processed_combo, frame_num)
                    if combo_val is not None:
                        last_combo_value = combo_val
                last_combo_hash = combo_hash
            current_data_point['combo'] = last_combo_value

            acc_hash = result.get('acc_hash')
            if acc_hash is not None and acc_hash != last_acc_hash:
                acc_img = result.get('acc_img')
                if acc_img is not None:
                    processed_acc = self.ocr_calibrator.preprocess_for_ocr(acc_img, 'accuracy',
                                                                           self.current_ocr_settings['accuracy'])
                    acc_val = self.recognition_system.recognize_accuracy(processed_acc, frame_num)
                    if acc_val is not None:
                        last_acc_value = acc_val
                last_acc_hash = acc_hash
            current_data_point['accuracy'] = last_acc_value

            self.analysis_data["data_points"].append(current_data_point)

        for circle_id in list(self.active_circles.keys()):
            vanished_circle = self.active_circles.pop(circle_id)
            self.completed_circles.append({
                "id": vanished_circle['id'], "x": int(vanished_circle['pos'][0]), "y": int(vanished_circle['pos'][1]),
                "start_ts": vanished_circle['start_ts'], "end_ts": vanished_circle['last_seen_ts'], "type": "circle"
            })

        self.analysis_data['hit_objects'] = self._normalize_circle_durations()

        print(f"\n--- Analysis Summary ---")
        print(f"  - Found a total of {total_raw_detections} raw circle detections across all frames.")
        print(f"  - Tracker completed {len(self.completed_circles)} circle lifecycles.")
        print(
            f"  - Final result contains {len(self.analysis_data['hit_objects'])} total hit objects with standardized durations.")
        print(f"------------------------")

    def _update_circle_tracker(self, detected_circles: List[Tuple[int, int, int]], timestamp: float):
        """Updates the state of tracked circles based on newly detected circles."""
        detected_centers = np.array([[c[0], c[1]] for c in detected_circles]) if detected_circles else np.empty((0, 2))
        unmatched_detections = list(range(len(detected_centers)))

        if self.active_circles and detected_circles:
            active_ids = list(self.active_circles.keys())
            active_centers = np.array([self.active_circles[id]['pos'] for id in active_ids])
            dist_matrix = np.linalg.norm(active_centers[:, np.newaxis, :] - detected_centers[np.newaxis, :, :], axis=2)

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
                    "start_ts": vanished_circle['start_ts'], "end_ts": vanished_circle['last_seen_ts'],
                    "type": "circle"
                })

        for detection_idx in unmatched_detections:
            new_pos = detected_centers[detection_idx]
            self.active_circles[self.next_circle_id] = {
                'id': self.next_circle_id, 'pos': [int(new_pos[0]), int(new_pos[1])],
                'start_ts': timestamp, 'last_seen_ts': timestamp, 'unseen_frames': 0
            }
            self.next_circle_id += 1
