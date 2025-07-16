import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any

# Local imports
from config_manager import ConfigManager
from recognition import OsuRecognitionSystem


class OCRCalibrator:
    """A tool for live OCR preprocessing calibration and preset management."""

    def __init__(self, video_path: str, config_manager: ConfigManager):
        self.video_path = video_path
        self.config_manager = config_manager
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.output_dir = Path("src/debug_output")

        self.calibration_data = self._load_calibration_data()
        self.ui_regions = self.calibration_data.get('ui_regions', {})

        self.recognition_system = OsuRecognitionSystem(debug_mode=True)
        self.current_ocr_settings: Dict[str, Any] = {}
        self._load_ocr_settings()

    def _load_calibration_data(self) -> Dict:
        config_path = self.output_dir / "calibration_data.json"
        if config_path.exists():
            with open(config_path, 'r') as f: return json.load(f)
        return {}

    def _load_ocr_settings(self):
        self.current_ocr_settings = {
            'combo': self.config_manager.get_default_combo_settings(),
            'accuracy': self.config_manager.get_default_accuracy_settings(),
            'default': self.config_manager.get_default_other_settings()
        }

    def get_region_image(self, frame: np.ndarray, region_name: str) -> Optional[np.ndarray]:
        region_coords = self.ui_regions.get(region_name)
        if region_coords and len(region_coords) == 4:
            x, y, w, h = region_coords
            if w > 0 and h > 0:
                return frame[y:y + h, x:x + w]
        return None

    def preprocess_for_ocr(self, image: np.ndarray, region_name: str,
                           specific_settings: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Apply preprocessing steps to an image region based on OCR settings."""
        if image.size == 0: return np.array([])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        settings = specific_settings or self.current_ocr_settings.get(region_name, {})

        if settings.get("invert", 0) == 1:
            gray = cv2.bitwise_not(gray)

        if settings.get("median_blur_ksize", 0) > 0:
            ksize = settings["median_blur_ksize"]
            gray = cv2.medianBlur(gray, ksize if ksize % 2 != 0 else ksize + 1)

        threshold_type = settings.get("threshold_type", "THRESHOLD")
        if threshold_type == "ADAPTIVE_THRESH_GAUSSIAN_C":
            block_size = settings.get("block_size", 11)
            if block_size <= 1: block_size = 3
            if block_size % 2 == 0: block_size += 1
            processed_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                  block_size, settings.get("C_value", 2))
        elif threshold_type == "ADAPTIVE_THRESH_MEAN_C":
            block_size = settings.get("block_size", 11)
            if block_size <= 1: block_size = 3
            if block_size % 2 == 0: block_size += 1
            processed_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size,
                                                  settings.get("C_value", 2))
        else:  # "THRESHOLD"
            _, processed_img = cv2.threshold(gray, settings.get("threshold_value", 127), 255, cv2.THRESH_BINARY)

        if settings.get("morph_close_kernel", 0) > 0:
            ksize = settings["morph_close_kernel"]
            kernel = np.ones((ksize, ksize), np.uint8)
            processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel)

        if settings.get("fill_contours", 0) == 1:
            contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(processed_img, contours, -1, (255), thickness=cv2.FILLED)

        if settings.get("dilate_kernel_size", 0) > 0:
            kernel = np.ones((settings["dilate_kernel_size"], settings["dilate_kernel_size"]), np.uint8)
            processed_img = cv2.dilate(processed_img, kernel, iterations=settings.get("dilate_iterations", 1))

        if settings.get("final_invert", 0) == 1:
            processed_img = cv2.bitwise_not(processed_img)

        return processed_img

    def run(self, region_name: str = 'combo'):
        """
        Provides a live preview for calibrating OCR settings in a single window
        with a custom-drawn interactive control panel. Starts with 'combo' by default.
        """
        print(f"\n--- Live OCR Calibration for Region: {region_name.upper()} ---")
        print("Adjust sliders/buttons. See results in the preview window.")
        print("Controls: 'm'/'n' (seek frames), 's' (save), 'l' (load), 'TAB' (switch region), 'ESC' (exit).")

        control_window = f"OCR Control Panel: {region_name}"
        cv2.namedWindow(control_window)
        cv2.moveWindow(control_window, 100, 100)

        current_frame_pos = self.total_frames // 3
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read initial frame for calibration.")
            return

        def get_controls_for_region(r_name, settings):
            threshold_type_setting = settings.get("threshold_type", "THRESHOLD")
            internal_values = ["THRESHOLD", "ADAPTIVE_THRESH_MEAN_C", "ADAPTIVE_THRESH_GAUSSIAN_C"]
            display_options = ["THRESHOLD", "ADAPTIVE_MEAN", "ADAPTIVE_GAUSS"]
            try:
                current_val_index = internal_values.index(threshold_type_setting)
            except ValueError:
                current_val_index = 0

            return {
                'fill_contours': {'type': 'choice', 'val': settings.get("fill_contours", 0), 'options': ["Off", "On"]},
                'morph_close_kernel': {'type': 'slider', 'val': settings.get("morph_close_kernel", 0), 'min': 0,
                                       'max': 31},
                'threshold_type': {'type': 'choice', 'val': current_val_index, 'options': display_options},
                'threshold_value': {'type': 'slider', 'val': settings.get("threshold_value", 127), 'min': 0,
                                    'max': 255},
                'median_blur_ksize': {'type': 'slider', 'val': settings.get("median_blur_ksize", 0), 'min': 0,
                                      'max': 31},
                'dilate_kernel_size': {'type': 'slider', 'val': settings.get("dilate_kernel_size", 0), 'min': 0,
                                       'max': 31},
                'dilate_iterations': {'type': 'slider', 'val': settings.get("dilate_iterations", 1), 'min': 1,
                                      'max': 10},
                'final_invert': {'type': 'choice', 'val': settings.get("final_invert", 0), 'options': ["Off", "On"]},
            }

        current_settings = self.current_ocr_settings.get(region_name,
                                                         self.config_manager.get_default_other_settings()).copy()
        controls = get_controls_for_region(region_name, current_settings)

        control_height = 35
        preview_width = 400
        y_offset_info = [0]

        def draw_controls(panel, controls_dict):
            panel_width = panel.shape[1]
            LABEL_WIDTH, BTN_WIDTH, VALUE_WIDTH, MARGIN = 130, 30, 40, 10
            dynamic_slider_width = max(20, panel_width - LABEL_WIDTH - (BTN_WIDTH * 2) - VALUE_WIDTH - (MARGIN * 5))
            y_offset = 0
            for name, ctrl in controls_dict.items():
                y_offset += control_height
                label_x, minus_btn_x = MARGIN, LABEL_WIDTH + MARGIN * 2
                slider_x_start = minus_btn_x + BTN_WIDTH + MARGIN
                plus_btn_x = slider_x_start + dynamic_slider_width + MARGIN
                value_x = plus_btn_x + BTN_WIDTH + MARGIN

                ctrl['minus_btn_region'] = (minus_btn_x, y_offset - 25, BTN_WIDTH, 30)
                ctrl['plus_btn_region'] = (plus_btn_x, y_offset - 25, BTN_WIDTH, 30)

                cv2.rectangle(panel, ctrl['minus_btn_region'][:2],
                              (ctrl['minus_btn_region'][0] + BTN_WIDTH, ctrl['minus_btn_region'][1] + 30), (80, 80, 80),
                              -1)
                cv2.putText(panel, "-", (minus_btn_x + 9, y_offset - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                            2)
                cv2.rectangle(panel, ctrl['plus_btn_region'][:2],
                              (ctrl['plus_btn_region'][0] + BTN_WIDTH, ctrl['plus_btn_region'][1] + 30), (80, 80, 80),
                              -1)
                cv2.putText(panel, "+", (plus_btn_x + 8, y_offset - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                            2)
                cv2.putText(panel, name.replace('_', ' ').title(), (label_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)

                if ctrl['type'] == 'slider':
                    ctrl['slider_region'] = (slider_x_start, y_offset - 15, dynamic_slider_width, 10)
                    cv2.rectangle(panel, ctrl['slider_region'][:2],
                                  (slider_x_start + dynamic_slider_width, y_offset - 5), (50, 50, 50), -1)
                    val_ratio = (ctrl['val'] - ctrl['min']) / (ctrl['max'] - ctrl['min']) if (ctrl['max'] - ctrl[
                        'min']) != 0 else 0
                    handle_x = int(slider_x_start + val_ratio * dynamic_slider_width)
                    cv2.rectangle(panel, (handle_x - 5, y_offset - 20), (handle_x + 5, y_offset), (200, 200, 200), -1)
                    cv2.putText(panel, str(ctrl['val']), (value_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)
                elif ctrl['type'] == 'choice':
                    cv2.putText(panel, ctrl['options'][ctrl['val']], (slider_x_start, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return panel

        settings_changed = True

        def mouse_callback(event, x, y, flags, param):
            nonlocal settings_changed
            adjusted_y = y - y_offset_info[0]

            def is_inside(px, py, rect):
                return rect[0] < px < rect[0] + rect[2] and rect[1] < py < rect[1] + rect[3]

            if event == cv2.EVENT_LBUTTONDOWN or (flags & cv2.EVENT_FLAG_LBUTTON):
                for name, ctrl in controls.items():
                    if is_inside(x, adjusted_y, ctrl['minus_btn_region']):
                        if event == cv2.EVENT_LBUTTONDOWN:
                            ctrl['val'] = (ctrl['val'] - 1) % len(ctrl['options']) if ctrl['type'] == 'choice' else max(
                                ctrl['min'], ctrl['val'] - 1)
                            settings_changed = True;
                            break
                    elif is_inside(x, adjusted_y, ctrl['plus_btn_region']):
                        if event == cv2.EVENT_LBUTTONDOWN:
                            ctrl['val'] = (ctrl['val'] + 1) % len(ctrl['options']) if ctrl['type'] == 'choice' else min(
                                ctrl['max'], ctrl['val'] + 1)
                            settings_changed = True;
                            break
                    elif ctrl.get('slider_region') and is_inside(x, adjusted_y, ctrl['slider_region']):
                        ratio = (x - ctrl['slider_region'][0]) / ctrl['slider_region'][2]
                        new_val = int(ctrl['min'] + ratio * (ctrl['max'] - ctrl['min']))
                        if new_val != ctrl['val']: ctrl['val'], settings_changed = new_val, True
                        break

        cv2.setMouseCallback(control_window, mouse_callback)

        while True:
            if settings_changed:
                for key, ctrl in controls.items():
                    current_settings[key] = ctrl['options'][ctrl['val']].replace('ADAPTIVE MEAN',
                                                                                 'ADAPTIVE_THRESH_MEAN_C').replace(
                        'ADAPTIVE GAUSS', 'ADAPTIVE_THRESH_GAUSSIAN_C') if key == 'threshold_type' else ctrl['val']
                    if 'ksize' in key and current_settings.get(key, 0) > 0 and current_settings.get(key, 0) % 2 == 0:
                        current_settings[key] += 1

                region_img = self.get_region_image(frame, region_name)
                if region_img is None or region_img.size == 0:
                    print(f"Region '{region_name}' not defined. Calibrate UI regions first.");
                    break

                processed_img = self.preprocess_for_ocr(region_img, region_name, specific_settings=current_settings)
                ocr_text = self.recognition_system.recognize_combo(processed_img,
                                                                   -1) if region_name == 'combo' else self.recognition_system.recognize_accuracy(
                    processed_img, -1)

                # --- LAYOUT CHANGE: Resize previews to be identical ---
                orig_h, orig_w = region_img.shape[:2]

                # Calculate the new height based on the original image's aspect ratio
                new_height = int(orig_h * (preview_width / orig_w)) if orig_w > 0 else 0
                target_size = (preview_width, new_height)

                # Resize both images to the exact same target size
                resized_orig = cv2.resize(region_img, target_size)
                resized_proc = cv2.resize(processed_img, target_size)

                # Convert the processed (grayscale) image to BGR for stacking
                resized_proc_bgr = cv2.cvtColor(resized_proc, cv2.COLOR_GRAY2BGR)

                # Stack the identically-sized previews horizontally
                preview_panel = np.hstack((resized_orig, resized_proc_bgr))
                total_width = preview_panel.shape[1]

                control_panel = np.zeros((len(controls) * control_height + 20, total_width, 3), dtype=np.uint8)
                control_panel = draw_controls(control_panel, controls)

                text_panel = np.zeros((40, total_width, 3), dtype=np.uint8)
                cv2.putText(text_panel, f"OCR: {ocr_text if ocr_text else 'None'}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                y_offset_info[0] = preview_panel.shape[0]
                composite_view = np.vstack((preview_panel, control_panel, text_panel))
                cv2.imshow(control_window, composite_view)
                settings_changed = False

            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            elif key == ord('s'):
                preset_name = input("\nEnter name to save this preset as: ").strip()
                if preset_name:
                    existing_preset = self.config_manager.get_preset(preset_name) or {}
                    existing_preset[region_name] = current_settings
                    self.config_manager.add_preset(preset_name, existing_preset)
            elif key == ord('l'):
                presets = self.config_manager.list_presets()
                if not presets: print("\nNo saved presets found."); continue
                print("\nAvailable presets:", ", ".join(presets))
                preset_to_load = input("Enter preset name to load: ").strip()
                loaded_data = self.config_manager.get_preset(preset_to_load)
                if loaded_data and region_name in loaded_data:
                    current_settings = loaded_data[region_name]
                    controls = get_controls_for_region(region_name, current_settings)
                    settings_changed = True
                    print(f"Loaded settings for '{region_name}' from preset '{preset_to_load}'.")
                else:
                    print(f"Preset '{preset_to_load}' not found or no settings for '{region_name}'.")
            elif key == 9:  # TAB
                region_name = 'accuracy' if region_name == 'combo' else 'combo'
                print(f"\nSwitched to calibrating region: {region_name.upper()}")
                cv2.setWindowTitle(control_window, f"OCR Control Panel: {region_name}")
                current_settings = self.current_ocr_settings.get(region_name,
                                                                 self.config_manager.get_default_other_settings()).copy()
                controls = get_controls_for_region(region_name, current_settings)
                settings_changed = True
            elif key == ord('m') or key == ord('n'):
                seek = 15 if key == ord('m') else -60
                current_frame_pos = max(0, min(self.total_frames - 1, current_frame_pos + seek))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                ret, frame = self.cap.read()
                if ret: settings_changed = True

        cv2.destroyAllWindows()
        print("Exited OCR calibration.")

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
