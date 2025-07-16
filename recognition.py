import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import os
from collections import deque
from typing import Optional, Dict


class OsuRecognitionSystem:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.ocr_config_general = '--psm 10'

        self.colors = {
            'combo_text': {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255])},
            'accuracy_text': {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255])},
            'miss_indicator': {'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255])},
            'hp_bar': {'lower': np.array([35, 50, 50]), 'upper': np.array([85, 255, 255])},
        }

        self.cursor_positions = deque(maxlen=100)

    def recognize_combo(self, combo_region: np.ndarray, frame_id: int) -> Optional[int]:
        """
        Recognizes the combo count from the given image region using OCR (Tesseract).
        Returns a cleaned integer, or None if recognition fails.
        """
        if combo_region.size == 0:
            return None

        pil_image = Image.fromarray(combo_region)
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789xX'
        try:
            text = pytesseract.image_to_string(pil_image, config=custom_config).strip()

            # Clean the text: remove anything that's not a digit
            cleaned_text = re.sub(r'[^0-9]', '', text)

            if cleaned_text:
                return int(cleaned_text)
            else:
                return None
        except (ValueError, TypeError):
            # ValueError if int() fails, TypeError for safety
            return None
        except Exception as e:
            if self.debug_mode:
                print(f"[Frame {frame_id}] Error recognizing combo: {e}")
            return None

    def recognize_accuracy(self, accuracy_region: np.ndarray, frame_id: int) -> Optional[str]:
        """
        Recognizes the accuracy percentage from the given image region.
        """
        if accuracy_region.size == 0:
            return None

        pil_image = Image.fromarray(accuracy_region)

        # OCR configuration for floating point numbers with a percentage sign
        # psm 8 for single word, psm 6 for single block of text
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.%'
        try:
            text = pytesseract.image_to_string(pil_image, config=custom_config).strip()

            cleaned_text = text.replace(' ', '').replace(',', '.')

            if not re.match(r'^\d+(\.\d+)?%?$', cleaned_text):
                if self.debug_mode:
                    print(f"[Frame {frame_id}] Accuracy clean failed: '{text}' -> '{cleaned_text}'")
                return None

            if re.match(r'^0(\.)?%?$', cleaned_text):
                if self.debug_mode:
                    None
                return None

            if re.match(r'^\d+(\.\d+)?$', cleaned_text) and '%' not in cleaned_text:
                cleaned_text += '%'

            return cleaned_text if cleaned_text else None
        except Exception as e:
            if self.debug_mode:
                print(f"[Frame {frame_id}] Error recognizing accuracy: {e}")
            return None


# Optional: Configure Tesseract path for Windows users if not in system PATH
if os.name == 'nt':  # For Windows
    try:
        # IMPORTANT: Change this path if your Tesseract installation is elsewhere
        pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    except pytesseract.TesseractNotFoundError:
        print("Tesseract is not installed or not found in the specified path.")
        print("Please install it from https://tesseract-ocr.github.io/tessdoc/Installation.html")
    except Exception as e:
        print(f"An unexpected error occurred while setting Tesseract path: {e}")
