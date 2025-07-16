import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

class ConfigManager:
    def __init__(self, config_dir="saves/overlay"):
        self.config_dir = Path(config_dir)
        os.makedirs(self.config_dir, exist_ok=True)
        self.presets_file = self.config_dir / "ocr_presets.json"
        self.presets: Dict[str, Dict[str, Any]] = self._load_presets()

    def _load_presets(self) -> Dict[str, Dict[str, Any]]:
        """Loads OCR presets from a JSON file."""
        if self.presets_file.exists():
            with open(self.presets_file, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {self.presets_file}. Starting with empty presets.")
                    return {}
        return {}

    def _save_presets(self):
        """Saves current OCR presets to the JSON file."""
        with open(self.presets_file, 'w') as f:
            json.dump(self.presets, f, indent=4)

    def add_preset(self, name: str, settings: Dict[str, Any]):
        """Adds or updates an OCR preset."""
        self.presets[name] = settings
        self._save_presets()
        print(f"Preset '{name}' saved.")

    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieves an OCR preset by name."""
        return self.presets.get(name)

    def list_presets(self) -> List[str]:
        """Returns a list of available preset names."""
        return list(self.presets.keys())

    def delete_preset(self, name: str) -> bool:
        """Deletes an OCR preset."""
        if name in self.presets:
            del self.presets[name]
            self._save_presets()
            print(f"Preset '{name}' deleted.")
            return True
        print(f"Preset '{name}' not found.")
        return False

    def get_default_combo_settings(self) -> Dict[str, Any]:
        """Returns default settings for combo OCR preprocessing."""
        return {
            "threshold_type": "ADAPTIVE_THRESH_GAUSSIAN_C",
            "threshold_mode": "THRESH_BINARY",
            "block_size": 41,
            "C_value": -105,
            "dilate_kernel_size": 8,
            "dilate_iterations": 1,
            "median_blur_ksize": 5
        }

    def get_default_accuracy_settings(self) -> Dict[str, Any]:
        """Returns default settings for accuracy OCR preprocessing."""
        return {
            "threshold_type": "ADAPTIVE_THRESH_GAUSSIAN_C",
            "threshold_mode": "THRESH_BINARY",
            "block_size": 31,
            "C_value": -35,
            "gaussian_blur_ksize": (3, 3)
        }

    def get_default_other_settings(self) -> Dict[str, Any]:
        """Returns default settings for general OCR preprocessing."""
        return {
            "threshold_type": "THRESHOLD",
            "threshold_mode": "THRESH_BINARY | THRESH_OTSU",
            "threshold_value": 150
        }
