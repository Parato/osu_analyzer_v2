import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict

# --- Constants for replay ---
APPROACH_CIRCLE_START_SCALE = 2.2


class Visualization:
    """Handles the creation of plots and interactive replays from analysis data."""

    def __init__(self, analysis_file_path: str):
        self.analysis_file_path = Path(analysis_file_path)
        if not self.analysis_file_path.exists():
            raise FileNotFoundError(f"Analysis file not found: {self.analysis_file_path}")

        with open(self.analysis_file_path, 'r') as f:
            self.analysis_data = json.load(f)

        self.video_info = self.analysis_data.get('video_info', {})
        self.video_path = self.video_info.get('path')
        self.width = self.video_info.get('width', 1920)
        self.height = self.video_info.get('height', 1080)
        self.fps = self.video_info.get('fps', 30)

        self.output_dir = Path("src/saves/overlay")

    def create_data_plot(self):
        """Creates and saves a plot of combo, accuracy, and the estimated star rating."""
        print("Generating analysis graphs...")

        data_points = self.analysis_data.get('data_points', [])
        star_rating = self.analysis_data.get('star_rating', 0.0)

        timestamps_combo = [dp['timestamp'] for dp in data_points if dp.get('combo') is not None]
        combo_values = [dp['combo'] for dp in data_points if dp.get('combo') is not None]

        timestamps_acc = [dp['timestamp'] for dp in data_points if dp.get('accuracy') is not None]
        acc_values = [float(str(dp['accuracy']).replace('%', '')) for dp in data_points if
                      dp.get('accuracy') is not None]

        if not timestamps_combo and not timestamps_acc:
            print("No combo or accuracy data available to generate graphs.")
            if star_rating == 0.0:
                return

        # Changed from 3 subplots to 2
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        # Added star rating to title
        fig.suptitle(f'osu! Gameplay Analysis - Estimated ★ {star_rating:.2f}', fontsize=16)

        # Plot Combo
        ax1.plot(timestamps_combo, combo_values, label='Combo', color='deepskyblue', marker='.', linestyle='-')
        ax1.set_ylabel('Combo Count')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.legend()

        # Plot Accuracy
        ax2.plot(timestamps_acc, acc_values, label='Accuracy', color='limegreen', marker='.', linestyle='-')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_xlabel('Time (s)') # Added X label here since it's the bottom plot
        if acc_values:
            ax2.set_ylim(min(acc_values) - 1 if min(acc_values) < 100 else 99, 100.5)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = "src/saves/result/analysis_plots.png"
        plt.savefig(save_path)
        print(f"Analysis plot saved to {save_path}")
        plt.show()

    def view_replay(self):
        """Creates an interactive replay with hit object lifecycles."""
        print("\nOpening Hit Object Replay...")
        if not self.video_path or not Path(self.video_path).exists():
            print("❌ Video path from analysis file not found. Cannot start replay.")
            return

        hit_objects = self.analysis_data.get('hit_objects', [])
        if not hit_objects:
            print("\n❌ No hit objects to display in the replay.")
            print("   This usually means the analysis did not detect any circles.")
            print(
                "   Try re-running the '2. Calibrate Hit Circles' step with different parameters or on a different part of the video.")
            return

        calib_path = self.output_dir / "calibration_data.json"
        if not calib_path.exists():
            print(f"❌ Calibration data not found at '{calib_path}'. Cannot determine circle size.")
            return
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
            hit_circle_radius = calib_data.get('hit_circle_params', {}).get('maxRadius')
            if not hit_circle_radius:
                print("❌ 'maxRadius' not found in calibration data. Cannot determine circle size.")
                return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file for replay: {self.video_path}")
            return

        replay_window = "Hit Object Replay"
        cv2.namedWindow(replay_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(replay_window, 1280, 720)

        all_objects = sorted(hit_objects, key=lambda c: c['start_ts'])
        if not all_objects:
            print("No objects to replay.");
            return

        max_time_ms = int(all_objects[-1]['end_ts'] * 1000) if all_objects else 1
        if max_time_ms <= 0: max_time_ms = 1

        is_playing = True
        current_time_ms = 0
        last_tick_count = cv2.getTickCount()
        programmatic_update = False
        video_opacity = 70

        def on_trackbar_change(val):
            nonlocal current_time_ms, is_playing, last_tick_count
            if not programmatic_update:
                current_time_ms = val
                is_playing = False
                last_tick_count = cv2.getTickCount()

        def on_opacity_change(val):
            nonlocal video_opacity
            video_opacity = val

        cv2.createTrackbar("Timeline", replay_window, 0, max_time_ms, on_trackbar_change)
        cv2.createTrackbar("Video Opacity", replay_window, video_opacity, 100, on_opacity_change)

        while True:
            if is_playing:
                now_ticks = cv2.getTickCount()
                delta_time_s = (now_ticks - last_tick_count) / cv2.getTickFrequency()
                last_tick_count = now_ticks
                current_time_ms += int(delta_time_s * 1000)

                programmatic_update = True
                cv2.setTrackbarPos("Timeline", replay_window, current_time_ms)
                programmatic_update = False

            current_time_s = current_time_ms / 1000.0
            video_alpha = video_opacity / 100.0

            if video_alpha > 0:
                frame_pos = int(current_time_s * self.fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, video_frame = cap.read()
                if not ret:
                    video_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                canvas = cv2.addWeighted(video_frame, video_alpha, np.zeros_like(video_frame), 1 - video_alpha, 0)
            else:
                canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            overlay = np.zeros_like(canvas, dtype=np.uint8)

            for obj in hit_objects:
                start_ts, end_ts = obj['start_ts'], obj['end_ts']

                if start_ts <= current_time_s < end_ts:
                    if obj['type'] == 'circle':
                        cv2.circle(canvas, (obj['x'], obj['y']), hit_circle_radius, (255, 255, 255), 2)

                    duration = end_ts - start_ts
                    if duration > 0:
                        progress = (current_time_s - start_ts) / duration
                        start_radius = int(hit_circle_radius * APPROACH_CIRCLE_START_SCALE)
                        current_radius = int(start_radius + (hit_circle_radius - start_radius) * progress)

                        pos = (obj['x'], obj['y'])
                        cv2.circle(overlay, pos, current_radius, (255, 0, 0), 2)

            final_frame = cv2.add(canvas, overlay)
            play_status = "Playing" if is_playing else "Paused"
            cv2.putText(final_frame, f"Time: {current_time_s:.2f}s / {max_time_ms / 1000.0:.2f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(final_frame, f"Status: {play_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(final_frame, "SPACE: Play/Pause | 'a'/'d': Seek 5s | ESC: Quit", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(replay_window, final_frame)
            key = cv2.waitKey(1 if is_playing else 30) & 0xFF

            if key == 27:
                break
            elif key == 32:
                is_playing = not is_playing
                if is_playing: last_tick_count = cv2.getTickCount()
            elif key == ord('a'):
                current_time_ms -= 5000
                last_tick_count = cv2.getTickCount()
            elif key == ord('d'):
                current_time_ms += 5000
                last_tick_count = cv2.getTickCount()

            current_time_ms = max(0, min(current_time_ms, max_time_ms))
            if key in [ord('a'), ord('d')]:
                programmatic_update = True
                cv2.setTrackbarPos("Timeline", replay_window, current_time_ms)
                programmatic_update = False

            if is_playing and current_time_ms >= max_time_ms:
                current_time_ms = 0
                last_tick_count = cv2.getTickCount()
                programmatic_update = True
                cv2.setTrackbarPos("Timeline", replay_window, current_time_ms)
                programmatic_update = False

        cap.release()
        cv2.destroyAllWindows()
