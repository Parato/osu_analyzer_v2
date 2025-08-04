# analysis_helpers.py
#
# Contains helper functions to analyze beatmap data and identify
# information-rich sections for dataset generation.

import math


def find_best_section_by_count(hit_objects, duration_ms, object_type_filter):
    """
    Finds a time window that contains the maximum number of a specific object type.

    Args:
        hit_objects (list): The list of all hit objects in the beatmap.
        duration_ms (int): The desired duration of the clip in milliseconds.
        object_type_filter (str): 'circle' or 'slider' to specify which objects to count.

    Returns:
        dict: A dictionary {'start': ms, 'end': ms, 'type': str} for the best section, or None.
    """
    max_count = 0
    best_start_time = 0

    if not hit_objects:
        return None

    is_circle_filter = object_type_filter == 'circle'

    # Iterate through each hit object as a potential start for our window
    for i, start_obj in enumerate(hit_objects):
        current_count = 0
        window_start_time = start_obj['time']
        window_end_time = window_start_time + duration_ms

        # Count relevant objects within this window
        for j in range(i, len(hit_objects)):
            obj_to_check = hit_objects[j]
            if obj_to_check['time'] > window_end_time:
                break

            is_slider = obj_to_check.get('is_slider', False)
            if (is_circle_filter and not is_slider) or (not is_circle_filter and is_slider):
                current_count += 1

        if current_count > max_count:
            max_count = current_count
            best_start_time = window_start_time

    if max_count > 0:
        return {'start': best_start_time, 'end': best_start_time + duration_ms, 'type': object_type_filter}
    return None


def get_spinner_clips(hit_objects):
    """
    Finds all spinners and creates clips with dynamic durations.
    The clip shows the full spinner + 5s before and 5s after.

    Args:
        hit_objects (list): The list of all hit objects in the beatmap.

    Returns:
        list: A list of clip dictionaries, one for each spinner.
    """
    spinner_clips = []
    for obj in hit_objects:
        if obj.get('is_spinner'):
            clip_start = max(0, obj['time'] - 2000)
            clip_end = obj['end_time'] + 2000
            spinner_clips.append({'start': int(clip_start), 'end': int(clip_end), 'type': 'spinner'})
    return spinner_clips


def get_universal_value_score(hit_objects, clip, slider_weight=1.5):
    """
    Calculates a value score for a clip based on its contents, weighting
    sliders more heavily as they are more visually complex.

    Args:
        hit_objects (list): The list of all hit objects in the beatmap.
        clip (dict): The clip dictionary {'start': ms, 'end': ms}.
        slider_weight (float): The multiplier for sliders.

    Returns:
        float: The calculated value score for the clip.
    """
    if not clip:
        return 0

    score = 0
    for obj in hit_objects:
        if clip['start'] <= obj['time'] < clip['end']:
            if obj.get('is_slider'):
                score += slider_weight
            else:
                score += 1.0  # Circles and spinners get a base score
    return score