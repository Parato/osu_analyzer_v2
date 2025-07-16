import cv2
import numpy as np
import subprocess
import shutil
import os
from pathlib import Path
from typing import Dict, Optional
import multiprocessing


def standardize_video_if_needed(video_path: str) -> Optional[str]:
    """
    Checks video properties and re-encodes to 1920x1080 @ 30fps, overwriting the original.
    """
    TARGET_WIDTH = 1920
    TARGET_HEIGHT = 1080
    TARGET_FPS = 30.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    is_standardized = (width == TARGET_WIDTH and height == TARGET_HEIGHT and abs(fps - TARGET_FPS) < 0.1)

    if is_standardized:
        print(f"Video '{os.path.basename(video_path)}' already meets the standard (1920x1080 @ 30fps).")
        return video_path

    p = Path(video_path)
    temp_output_path = p.with_name(f"{p.stem}_temp_conversion.mp4")

    print("\n" + "=" * 50)
    print(f"Video '{p.name}' is not standard.")
    print(f"  - Current: {width}x{height} @ {fps:.2f} FPS")
    print(f"  - Target:  {TARGET_WIDTH}x{TARGET_HEIGHT} @ {TARGET_FPS:.2f} FPS")
    print(f"Standardizing video... The original file will be overwritten.")
    print("=" * 50)

    ffmpeg_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'scale={TARGET_WIDTH}:{TARGET_HEIGHT},fps={TARGET_FPS}',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '22',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-y',  # Overwrite temp file if it exists
        str(temp_output_path)
    ]

    try:
        process = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        # Replace original file with the new one
        shutil.move(str(temp_output_path), video_path)
        print(f"✓ Standardization complete. '{p.name}' has been overwritten.")
        return video_path
    except subprocess.CalledProcessError as e:
        print("❌ FFmpeg conversion failed.")
        print("FFmpeg stderr:", e.stderr)
        if temp_output_path.exists():
            os.remove(temp_output_path)
        return None
    except FileNotFoundError:
        print("❌ FFmpeg not found. Please ensure it is installed and in your system's PATH.")
        return None
    except Exception as e:
        print(f"An error occurred during file replacement: {e}")
        if temp_output_path.exists():
            os.remove(temp_output_path)
        return None


def detection_worker(video_path: str, ui_regions: Dict, hit_circle_params: Dict, frame_queue: multiprocessing.Queue,
                     result_queue: multiprocessing.Queue):
    """
    A worker process that takes frames from a queue, performs detection, and puts results in another queue.
    This function is designed to be stateless and only perform detection on a single frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    while True:
        frame_num = frame_queue.get()
        if frame_num is None:  # Sentinel value to stop the worker
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        result = {'frame': frame_num, 'circles': [], 'combo_hash': None, 'acc_hash': None, 'combo_img': None,
                  'acc_img': None}

        # --- Hit Circle Detection with Completeness Filter ---
        if hit_circle_params:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            p = hit_circle_params

            detected_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                                param1=p.get('param1', 50), param2=p.get('param2', 30),
                                                minRadius=p.get('minRadius', 10), maxRadius=p.get('maxRadius', 50))

            if detected_circles is not None:
                completeness_thresh_percent = p.get('completeness')

                # If completeness is not set in config, or if it's set to >=100, disable the filter
                if completeness_thresh_percent is None or completeness_thresh_percent >= 100:
                    result['circles'] = np.uint16(np.around(detected_circles))[0, :].tolist()
                else:
                    completeness_thresh = completeness_thresh_percent / 100.0
                    canny_edges = cv2.Canny(gray, 50, 150)
                    valid_circles = []

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
                            valid_circles.append(c)

                    if valid_circles:
                        result['circles'] = np.uint16(np.around(valid_circles)).tolist()

        # --- Selective OCR Hashing ---
        combo_coords = ui_regions.get('combo')
        if combo_coords:
            x, y, w, h = combo_coords
            combo_img = frame[y:y + h, x:x + w]
            if combo_img.size > 0:
                result['combo_img'] = combo_img
                result['combo_hash'] = np.mean(combo_img)

        acc_coords = ui_regions.get('accuracy')
        if acc_coords:
            x, y, w, h = acc_coords
            acc_img = frame[y:y + h, x:x + w]
            if acc_img.size > 0:
                result['acc_img'] = acc_img
                result['acc_hash'] = np.mean(acc_img)

        result_queue.put(result)

    cap.release()
