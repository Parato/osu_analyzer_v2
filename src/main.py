#!/usr/bin/env python3
"""
osu! Video Analysis Tool Launcher
Provides a simple interface to calibrate and analyze osu! gameplay videos.
"""

import os
import sys
import json
import shutil
import multiprocessing
from pathlib import Path
from typing import Optional

# Local imports
import video_processing
import analysis
import visualization
from config_manager import ConfigManager
# --- New imports for specialized calibrators ---
from ui_calibrator import UICalibrator
from hit_object_calibrator import HitObjectCalibrator
from ocr_calibrator import OCRCalibrator


class OsuAnalysisLauncher:
    """Main class to launch and manage the analysis tool."""

    def __init__(self):
        self.current_dir = Path("")
        self.config_file = Path("src/saves/overlay") / "calibration_data.json"
        self.analysis_file = Path("src/saves/result") / "analysis_debug.json"
        self.config_manager = ConfigManager()

    def find_videos(self) -> list[str]:
        """Find video files in common directories."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.flv', '.webm']
        search_dirs = ["src/test_videos"]

        found_videos = []
        for directory in search_dirs:
            dir_path = Path(directory)
            if dir_path.exists():
                for file_path in dir_path.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                        found_videos.append(str(file_path))
        return found_videos

    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        print("Checking dependencies...")
        all_ok = True
        required_modules = {
            'cv2': 'opencv-python',
            'numpy': 'numpy',
            'pytesseract': 'pytesseract',
            'matplotlib': 'matplotlib'
        }

        try:
            import cv2.ximgproc
        except ImportError:
            all_ok = False
            print("❌ Missing required package for slider path detection: opencv-contrib-python")
            print("   Please install it by running: pip install opencv-contrib-python")

        for module, package in required_modules.items():
            try:
                __import__(module)
            except ImportError:
                all_ok = False
                print(f"❌ Missing Python package: {package}")
                print(f"   Install with: pip install {package}")

        if not shutil.which("ffmpeg"):
            all_ok = False
            print("❌ FFmpeg not found. Please install FFmpeg and ensure it's in your system's PATH.")
            print("   Download from: https://ffmpeg.org/download.html")
        else:
            print("✓ FFmpeg found.")
        return all_ok

    def show_configuration_status(self):
        """Display a brief summary of the current configuration status."""
        print("Configuration Status:")
        if not self.config_file.exists():
            print("  ❌ No configuration file found. Please run calibration.")
            return

        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)

            if config.get('ui_regions', {}).get('combo') and config.get('ui_regions', {}).get('accuracy'):
                print("  ✓ UI regions configured.")
            else:
                print("  ❌ UI regions NOT configured.")

            if config.get('hit_circle_params'):
                print("  ✓ Hit Objects (Circles) configured.")
            else:
                print("  ❌ Hit Objects NOT configured.")

        except Exception as e:
            print(f"  ❌ Error reading configuration: {e}")

    def select_video(self) -> Optional[str]:
        """Interactive video selection."""
        videos = self.find_videos()
        if not videos:
            print("❌ No videos found in common directories")
            video_path = input("Enter video path manually: ").strip()
            if not os.path.exists(video_path):
                print("❌ Video file not found")
                return None
            return video_path

        print("\nFound videos:")
        for i, video in enumerate(videos, 1):
            print(f"{i}. {video}")

        try:
            choice = int(input("\nEnter video number: ")) - 1
            if 0 <= choice < len(videos):
                return videos[choice]
            else:
                print("❌ Invalid choice")
                return None
        except (ValueError, KeyboardInterrupt):
            print("Cancelled")
            return None

    def run(self):
        """Run the launcher, check dependencies, and start the main menu."""
        print("osu! Video Analysis Tool")
        print("========================")

        if not self.check_dependencies():
            return

        self.main_menu()

    def main_menu(self):
        """Main menu interface."""
        while True:
            print("\n" + "=" * 50)
            print("OSU ANALYZER MENU")
            print("=" * 50)
            self.show_configuration_status()
            print("\nOptions:")
            print("1. Calibrate UI Regions (Combo, Accuracy)")
            print("2. Calibrate Hit Objects (Circles)")
            print("3. Live OCR Preview & Create/Edit Preset")
            print("4. Analyze Video")
            print("5. View Hit Object Replay")
            print("6. Manage OCR Presets")
            print("7. Check Dependencies")
            print("8. Exit")

            try:
                choice = input("\nEnter your choice (1-8): ").strip()

                if choice in ['1', '2', '3']:
                    video_path = self.select_video()
                    if not video_path: continue
                    std_path = video_processing.standardize_video_if_needed(video_path)
                    if not std_path: continue

                    if choice == '1':
                        calibrator = UICalibrator(std_path, self.config_manager)
                        calibrator.run()
                    elif choice == '2':
                        calibrator = HitObjectCalibrator(std_path, self.config_manager)
                        calibrator.run()
                    elif choice == '3':
                        calibrator = OCRCalibrator(std_path, self.config_manager)
                        calibrator.run()

                elif choice == '4':
                    video_path = self.select_video()
                    if not video_path: continue
                    std_path = video_processing.standardize_video_if_needed(video_path)
                    if not std_path: continue
                    preset_name = self.select_ocr_preset()
                    engine = analysis.AnalysisEngine(std_path, preset_name, self.config_manager)
                    engine.run_full_analysis()
                elif choice == '5':
                    if not self.analysis_file.exists():
                        print("\n❌ No analysis data found. Please run '4. Analyze video' first.")
                        continue
                    vis = visualization.Visualization(self.analysis_file)
                    vis.view_replay()
                elif choice == '6':
                    self.manage_ocr_presets_menu()
                elif choice == '7':
                    self.check_dependencies()
                elif choice == '8':
                    print("Goodbye!")
                    break
                else:
                    print("❌ Invalid choice")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred in the main menu: {e}")
                import traceback
                traceback.print_exc()

    def manage_ocr_presets_menu(self):
        """Menu for managing OCR presets."""
        while True:
            print("\n" + "=" * 40)
            print("Manage OCR Presets")
            print("=" * 40)
            presets = self.config_manager.list_presets()
            if not presets:
                print("No OCR presets found.")
            else:
                print("Available Presets:")
                for i, name in enumerate(presets, 1):
                    print(f"{i}. {name}")
            print("\nOptions:")
            print("1. Create/Edit Preset (via Live Preview)")
            print("2. Delete Preset")
            print("3. Back to Main Menu")

            preset_choice = input("Enter your choice (1-3): ").strip()

            if preset_choice == '1':
                print("Please use option '3. Live OCR Preview...' from the main menu.")
                break
            elif preset_choice == '2':
                if not presets:
                    print("No presets to delete.")
                    continue
                try:
                    delete_idx = int(input("Enter preset number to delete: ")) - 1
                    if 0 <= delete_idx < len(presets):
                        self.config_manager.delete_preset(presets[delete_idx])
                    else:
                        print("Invalid preset number.")
                except ValueError:
                    print("Invalid input.")
            elif preset_choice == '3':
                break
            else:
                print("Invalid choice.")

    def select_ocr_preset(self) -> Optional[str]:
        """Allows user to select an OCR preset for analysis."""
        presets = self.config_manager.list_presets()
        if not presets:
            print("No custom OCR presets found. Using default settings for analysis.")
            return None

        print("\nSelect OCR Preset for Analysis:")
        print("0. Use Default OCR Settings")
        for i, name in enumerate(presets, 1):
            print(f"{i}. {name}")

        try:
            choice = int(input("Enter preset number (0 for default): "))
            if choice == 0:
                print("Using default settings.")
                return None
            elif 1 <= choice <= len(presets):
                selected_preset = presets[choice - 1]
                print(f"Using preset: '{selected_preset}'")
                return selected_preset
            else:
                print("Invalid choice. Using default settings.")
                return None
        except ValueError:
            print("Invalid input. Using default settings.")
            return None


def main():
    """Entry point"""
    multiprocessing.freeze_support()
    launcher = OsuAnalysisLauncher()
    launcher.run()


if __name__ == "__main__":
    main()
