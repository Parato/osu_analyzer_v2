# osu_parser.py
#
# Contains the logic for parsing .osu beatmap files.
# UPDATED: Added Hard Rock (HR) and Double Time (DT) modification logic.
# FIXED: Resolved final type warnings by using float literals in HR difficulty calculations.
# MODIFIED: Now parses StackLeniency from the [General] section.
# FIXED: Removed HR coordinate flipping from the parser to fix a bug. This is now handled in the dataset generator.

from utils import print_status
import numpy as np
import bezier


def get_timing_windows(od):
    """
    Calculates the hit windows in milliseconds for 300, 100, and 50 hits
    based on the Overall Difficulty (OD). Returns integer values.

    Args:
        od (float): The Overall Difficulty value of the beatmap.

    Returns:
        dict: A dictionary containing the rounded, integer timing windows in ms.
    """
    return {
        '300': round(80 - 6 * od),
        '100': round(140 - 8 * od),
        '50': round(200 - 10 * od)
    }


def get_ar_ms(ar):
    """Converts Approach Rate (AR) to milliseconds."""
    if ar < 5:
        return 1200 + 600 * (5 - ar) / 5
    elif ar == 5:
        return 1200
    else:
        return 1200 - 750 * (ar - 5) / 5


def ar_ms_to_val(ar_ms):
    """Converts AR milliseconds back to an AR value."""
    if ar_ms > 1200:
        return 5.0 - 5.0 * (ar_ms - 1200.0) / 600.0
    elif ar_ms < 1200:
        return 5.0 + 5.0 * (1200.0 - ar_ms) / 750.0
    else:
        return 5.0


def od_ms_to_val(od_300_ms):
    """Converts OD milliseconds (300-window) back to an OD value."""
    return (80.0 - od_300_ms) / 6.0


def parse_beatmap(beatmap_path, apply_hr=False, apply_dt=False):
    """
    Parses the .osu file to extract hit object, difficulty, and timing data.
    Optionally applies Hard Rock (HR) and Double Time (DT) modifications.
    """
    print_status(f"Parsing beatmap: {beatmap_path}")

    hit_objects = []
    difficulty = {}
    timing_points = []
    combo_colors = []
    timing_windows = {}

    in_general_section = False
    in_difficulty_section = False
    in_timing_points_section = False
    in_colors_section = False
    in_hit_objects_section = False

    try:
        with open(beatmap_path, 'r', encoding='utf-8') as f:
            combo_number = 1
            for line in f:
                line = line.strip()
                if not line: continue

                # Section handling
                if line == '[General]':
                    in_general_section, in_difficulty_section, in_timing_points_section, in_colors_section, in_hit_objects_section = True, False, False, False, False
                    continue
                elif line == '[Difficulty]':
                    in_general_section, in_difficulty_section, in_timing_points_section, in_colors_section, in_hit_objects_section = False, True, False, False, False
                    continue
                elif line == '[TimingPoints]':
                    in_general_section, in_difficulty_section, in_timing_points_section, in_colors_section, in_hit_objects_section = False, False, True, False, False
                    continue
                elif line == '[Colours]':
                    in_general_section, in_difficulty_section, in_timing_points_section, in_colors_section, in_hit_objects_section = False, False, False, True, False
                    continue
                elif line == '[HitObjects]':
                    in_general_section, in_difficulty_section, in_timing_points_section, in_colors_section, in_hit_objects_section = False, False, False, False, True
                    continue
                elif line.startswith('['):
                    in_general_section, in_difficulty_section, in_timing_points_section, in_colors_section, in_hit_objects_section = False, False, False, False, False
                    continue

                # Data parsing based on section
                if in_general_section:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        if key == 'StackLeniency':
                            difficulty['StackLeniency'] = float(parts[1].strip())

                elif in_difficulty_section:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = float(parts[1].strip())
                        difficulty[key] = value

                elif in_timing_points_section:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        timing_points.append({
                            'time': int(float(parts[0])),
                            'beat_length': float(parts[1]),
                            'uninherited': len(parts) > 6 and int(parts[6]) == 1
                        })

                elif in_colors_section:
                    parts = line.split(':')
                    if len(parts) == 2 and parts[0].strip().startswith('Combo'):
                        rgb_parts = parts[1].strip().split(',')
                        if len(rgb_parts) == 3:
                            combo_colors.append(tuple(int(p.strip()) for p in rgb_parts))

                elif in_hit_objects_section:
                    parts = line.split(',')
                    if len(parts) >= 5:
                        obj_type_int = int(parts[3])
                        is_new_combo = bool((obj_type_int >> 2) & 1)

                        if is_new_combo:
                            combo_number = 1

                        obj = {
                            'x': int(parts[0]), 'y': int(parts[1]),
                            'time': int(float(parts[2])),
                            'type': obj_type_int,
                            'new_combo': is_new_combo,
                            'is_slider': bool((obj_type_int >> 1) & 1),
                            'is_spinner': bool((obj_type_int >> 3) & 1),
                            'combo_number': combo_number,
                            'hit': False
                        }

                        if obj['is_slider']:
                            obj['slides'] = int(parts[6])
                            obj['length'] = float(parts[7])
                            try:
                                curve_points_str = parts[5]
                                points_str_list = curve_points_str.split('|')
                                control_points = [(obj['x'], obj['y'])]
                                for p_str in points_str_list[1:]:
                                    p_parts = p_str.split(':')
                                    control_points.append((int(p_parts[0]), int(p_parts[1])))
                                nodes = np.asfortranarray(control_points).T
                                obj['curve'] = bezier.Curve(nodes, degree=len(control_points) - 1)
                            except Exception as e:
                                print_status(f"Could not parse slider curve for object at time {obj['time']}: {e}",
                                             level="WARN")
                                obj['curve'] = None
                        elif obj['is_spinner']:
                            obj['end_time'] = int(float(parts[5]))

                        hit_objects.append(obj)
                        combo_number += 1

        if 'StackLeniency' not in difficulty:
            difficulty['StackLeniency'] = 0.7

        if apply_hr:
            print_status("Applying Hard Rock (HR) modifications to beatmap data...", level="INFO")

            difficulty['CircleSize'] = min(10.0, difficulty.get('CircleSize', 4) * 1.3)
            difficulty['ApproachRate'] = min(10.0, difficulty.get('ApproachRate', 9) * 1.4)
            difficulty['OverallDifficulty'] = min(10.0, difficulty.get('OverallDifficulty', 8) * 1.4)
            difficulty['HPDrainRate'] = min(10.0, difficulty.get('HPDrainRate', 5) * 1.4)
            print_status(f"HR difficulty: {difficulty}")

        if apply_dt:
            print_status("Applying Double Time (DT) modifications to beatmap data...", level="INFO")
            dt_rate = 1.5

            # Modify time values for all objects and timing points
            for obj in hit_objects:
                obj['time'] = obj['time'] / dt_rate
                if obj.get('is_spinner'):
                    obj['end_time'] = obj['end_time'] / dt_rate

            for tp in timing_points:
                tp['time'] = tp['time'] / dt_rate
                # Only inherited points (negative beat_length) are not affected by DT rate
                if tp['beat_length'] > 0:
                    tp['beat_length'] = tp['beat_length'] / dt_rate

            # Recalculate AR and OD
            ar = difficulty.get('ApproachRate', 9)
            od = difficulty.get('OverallDifficulty', 8)

            ar_ms = get_ar_ms(ar) / dt_rate
            od_300_ms = (80.0 - 6.0 * od) / dt_rate

            difficulty['ApproachRate'] = min(11.0, ar_ms_to_val(ar_ms))
            difficulty['OverallDifficulty'] = min(11.0, od_ms_to_val(od_300_ms))
            print_status(f"DT difficulty: {difficulty}")

        # Final calculation for timing windows based on the final OD
        if 'OverallDifficulty' in difficulty:
            timing_windows = get_timing_windows(difficulty['OverallDifficulty'])

        print_status(
            f"Successfully parsed {len(hit_objects)} hit objects, {len(timing_points)} timing points, and {len(combo_colors)} combo colors.")
        return hit_objects, difficulty, timing_points, combo_colors, timing_windows

    except FileNotFoundError:
        print_status(f"Beatmap file not found at: {beatmap_path}", level="ERROR")
        return None, None, None, None, None
    except Exception as e:
        print_status(f"An error occurred while parsing beatmap: {e}", level="ERROR")
        return None, None, None, None, None