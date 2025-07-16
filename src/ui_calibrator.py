import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Optional

# Local imports
from config_manager import ConfigManager


class UICalibrator:
    """A tool for interactively calibrating UI regions (combo, accuracy)."""

    def __init__(self, video_path: str, config_manager: ConfigManager):
        self.video_path = video_path
        self.config_manager = config_manager
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.output_dir = Path("src/saves/overlay")
        os.makedirs(self.output_dir, exist_ok=True)

        self.calibration_data = self._load_calibration_data()
        self.ui_regions = self.calibration_data.get('ui_regions', {})

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
        config_path = self.output_dir / "calibration_data.json"
        with open(config_path, 'w') as f:
            json.dump(self.calibration_data, f, indent=4)
        print(f"Calibration data saved to {config_path}")

    def run(self):
        """Starts the interactive UI calibration tool."""
        print("\nStarting interactive UI region calibration.")
        print("Instructions:")
        print("  - Click and drag to draw a box for the current region.")
        print("  - Press ENTER or SPACE to confirm and move to the next.")
        print("  - Press 'm'/'n' to skip frames, ESC to quit.")

        window_name = "UI Region Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        current_frame_pos = self.total_frames // 3
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read a frame for calibration.")
            return

        regions_to_calibrate = ["combo", "accuracy"]
        temp_regions = self.ui_regions.copy()

        for region_name in regions_to_calibrate:
            print(f"\nCalibrating region: {region_name.upper()}")
            current_roi = temp_regions.get(region_name)
            drawing = False
            start_point = None

            def mouse_callback(event, x, y, flags, param):
                nonlocal drawing, start_point, current_roi
                x = max(0, min(x, self.width - 1))
                y = max(0, min(y, self.height - 1))
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    start_point = (x, y)
                elif event == cv2.EVENT_MOUSEMOVE and drawing:
                    if start_point:
                        current_roi = (start_point[0], start_point[1], x - start_point[0], y - start_point[1])
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    if start_point:
                        new_x, new_y = min(start_point[0], x), min(start_point[1], y)
                        new_w, new_h = abs(x - start_point[0]), abs(y - start_point[1])
                        current_roi = (new_x, new_y, new_w, new_h)
                        print(f"  Selection made: {current_roi}. Press ENTER/SPACE to confirm.")

            cv2.setMouseCallback(window_name, mouse_callback)

            region_calibrated = False
            while not region_calibrated:
                display_frame = frame.copy()
                for r_name, r_coords in temp_regions.items():
                    if r_coords and r_name != region_name:
                        rx, ry, rw, rh = r_coords
                        cv2.rectangle(display_frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
                        cv2.putText(display_frame, r_name, (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                cv2.putText(display_frame, f"Calibrating: {region_name.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 0), 2)
                if current_roi:
                    x, y, w, h = current_roi
                    start_p = (x, y)
                    end_p = (x + w, y + h)
                    cv2.rectangle(display_frame, start_p, end_p, (0, 255, 0), 2)

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(20) & 0xFF

                if key == 27:
                    print("Calibration aborted.")
                    cv2.destroyAllWindows()
                    return
                elif key in [13, 32]:
                    if current_roi and current_roi[2] > 0 and current_roi[3] > 0:
                        temp_regions[region_name] = current_roi
                        print(f"  {region_name.upper()} region confirmed: {current_roi}")
                        region_calibrated = True
                    else:
                        print("  No valid region selected.")
                # --- BUG FIX: Add frame skipping with 'm' and 'n' ---
                elif key == ord('m'):
                    current_frame_pos = min(self.total_frames - 1, current_frame_pos + 30) # Skip 1 second
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                    ret, frame = self.cap.read()
                    if not ret: print("End of video reached.")
                elif key == ord('n'):
                    current_frame_pos = max(0, current_frame_pos - 30) # Go back 1 second
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                    ret, frame = self.cap.read()
                    if not ret: print("Start of video reached.")
                # --- END BUG FIX ---


        self.ui_regions = temp_regions
        self.calibration_data['ui_regions'] = self.ui_regions
        self._save_calibration_data()
        cv2.destroyAllWindows()

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
