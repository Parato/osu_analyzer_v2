import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict

# Local imports
from config_manager import ConfigManager


class HitObjectCalibrator:
    """A tool for interactively calibrating hit object detection parameters."""

    def __init__(self, video_path: str, config_manager: ConfigManager):
        self.video_path = video_path
        self.config_manager = config_manager
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.output_dir = Path("src/saves/overlay")
        os.makedirs(self.output_dir, exist_ok=True)

        self.calibration_data = self._load_calibration_data()
        self.hit_circle_params = self.calibration_data.get('hit_circle_params', {})

    def _load_calibration_data(self) -> Dict:
        """Loads calibration data from the config file."""
        config_path = self.output_dir / "calibration_data.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Could not decode JSON from {config_path}.")
        return {}

    def _save_calibration_data(self):
        """Saves the current calibration data."""
        # Clean up parameters from older versions before saving
        if 'slider_params' in self.calibration_data:
            del self.calibration_data['slider_params']
        if 'sliderFilter' in self.hit_circle_params:
            del self.hit_circle_params['sliderFilter']

        config_path = self.output_dir / "calibration_data.json"
        with open(config_path, 'w') as f:
            json.dump(self.calibration_data, f, indent=4)
        print(f"Calibration data saved to {config_path}")

    def run_circle_calibration(self):
        """Starts the interactive hit circle calibration tool."""
        print("\n--- Calibrating Hit Circles ---")
        print("INFO: Use the 'Circle Fullness' trackbar to filter out slider ends (half-circles).")
        print("      A higher value requires circles to be more complete. A good start is 70-80%.")

        window_name = "Hit Circle Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        current_frame_pos = self.total_frames // 3
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
        ret, frame = self.cap.read()
        if not ret:
            print("Error reading frame for calibration.")
            return

        p = self.hit_circle_params
        cv2.createTrackbar("Edge Threshold", window_name, p.get('param1', 50), 255, lambda v: v)
        cv2.createTrackbar("Accum Threshold", window_name, p.get('param2', 30), 100, lambda v: v)
        cv2.createTrackbar("Min Radius", window_name, p.get('minRadius', 20), 200, lambda v: v)
        cv2.createTrackbar("Max Radius", window_name, p.get('maxRadius', 60), 200, lambda v: v)
        cv2.createTrackbar("Circle Fullness %", window_name, p.get('completeness', 80), 100, lambda v: v)

        while True:
            display_frame = frame.copy()
            p1 = cv2.getTrackbarPos("Edge Threshold", window_name)
            p2 = cv2.getTrackbarPos("Accum Threshold", window_name)
            min_r = cv2.getTrackbarPos("Min Radius", window_name)
            max_r = cv2.getTrackbarPos("Max Radius", window_name)
            completeness_thresh_percent = cv2.getTrackbarPos("Circle Fullness %", window_name)
            completeness_thresh = completeness_thresh_percent / 100.0

            p1, p2, min_r = max(1, p1), max(1, p2), max(1, min_r)
            if max_r <= min_r:
                max_r = min_r + 1
                cv2.setTrackbarPos("Max Radius", window_name, max_r)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)

            detected_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                                param1=p1, param2=p2, minRadius=min_r, maxRadius=max_r)

            if detected_circles is not None:
                canny_edges = cv2.Canny(gray, 50, 150)
                circles_to_draw = []

                for c in detected_circles[0, :]:
                    center = (int(c[0]), int(c[1]))
                    radius = int(c[2])

                    circumference_mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(circumference_mask, center, radius, 255, 2)

                    total_circumference_pixels = np.count_nonzero(circumference_mask)
                    if total_circumference_pixels == 0: continue

                    edge_overlap_mask = cv2.bitwise_and(canny_edges, circumference_mask)
                    actual_edge_pixels = np.count_nonzero(edge_overlap_mask)

                    completeness_score = actual_edge_pixels / total_circumference_pixels

                    if completeness_score >= completeness_thresh:
                        circles_to_draw.append(c)

                if circles_to_draw:
                    circles_np = np.uint16(np.around(circles_to_draw))
                    for i in circles_np:
                        center, radius = (i[0], i[1]), i[2]
                        cv2.circle(display_frame, center, radius, (255, 0, 255), 2)
                        cv2.drawMarker(display_frame, center, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10,
                                       thickness=1)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                break
            elif key in [13, 32]:
                self.hit_circle_params = {
                    'param1': p1, 'param2': p2,
                    'minRadius': min_r, 'maxRadius': max_r,
                    'completeness': completeness_thresh_percent
                }
                self.calibration_data['hit_circle_params'] = self.hit_circle_params
                self._save_calibration_data()
                print("Hit circle parameters saved.")
                break
            elif key == ord('m'):
                current_frame_pos = min(self.total_frames - 1, current_frame_pos + 3)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                ret, frame = self.cap.read()
            elif key == ord('n'):
                current_frame_pos = max(0, current_frame_pos - 3)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                ret, frame = self.cap.read()

        cv2.destroyAllWindows()

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
