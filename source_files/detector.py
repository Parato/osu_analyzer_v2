# detector.py
#
# Contains the logic for detecting objects in a single frame.
# This version uses the 'ultralytics' library to run a trained YOLOv8 model.
# MODIFIED: Now filters detections using class-specific confidence thresholds.
# MODIFIED: Disabled verbose model output during prediction.

import config
import os
from ultralytics import YOLO


class CircleDetector:
    """A class to detect circles using a trained YOLOv8 .pt model."""

    def __init__(self):
        """
        Initializes the detector by loading the YOLOv8 model and the
        class-specific confidence thresholds.
        """
        print("[INFO] Loading YOLOv8 model from disk...")

        self.thresholds = config.YOLO_CONFIDENCE_THRESHOLDS
        self.default_threshold = self.thresholds.get('default', 0.25)

        if not os.path.exists(config.YOLO_MODEL_PATH):
            print(f"[WARN] Model file not found at '{config.YOLO_MODEL_PATH}'. The detector will not work.")
            print("[WARN] Please train a model and update the path in 'config.py'.")
            self.model = None
        else:
            self.model = YOLO(config.YOLO_MODEL_PATH)
            print("[INFO] YOLOv8 model loaded successfully.")
            print(f"[INFO] Using class-specific confidence thresholds. Default: {self.default_threshold}")

    def detect(self, frame):
        """
        Detects objects in a frame and filters them based on class-specific
        confidence thresholds.

        Args:
            frame (numpy.ndarray): The input video frame.

        Returns:
            list: A list of dictionaries representing valid, filtered detections.
        """
        if self.model is None:
            return []

        # Predict with a low global threshold to get all candidates.
        # --- FIX: Set verbose=False to disable the detailed per-frame console output. ---
        results = self.model.predict(frame, conf=0.10, verbose=False)

        result = results[0]
        filtered_detections = []

        for box in result.boxes:
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            confidence = float(box.conf)

            threshold = self.thresholds.get(class_name, self.default_threshold)

            if confidence >= threshold:
                xywh = box.xywh[0]
                x, y, w, h = map(int, xywh)

                filtered_detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "box": [x - w // 2, y - h // 2, w, h]
                })

        return filtered_detections