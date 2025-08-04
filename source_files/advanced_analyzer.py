# advanced_analyzer.py
#
# Contains the logic for analyzing events from the tracker and detector
# using beatmap data as a ground truth for high-fidelity analysis.

import math
# --- MODIFIED: No longer need print_status here, but keeping utils for get_center if it were moved ---
from utils import print_status
import config_generator as cfg


def get_center(box):
    """Calculates the center of a bounding box."""
    x, y, w, h = box
    return x + w // 2, y + h // 2


def get_spinner_req_rotations(od, duration_ms):
    """Calculates the number of rotations required to clear a spinner."""
    if od < 5:
        scaler = 3 + (5 - od) * 2 / 5
    elif od == 5:
        scaler = 5
    else:
        scaler = 5 + (od - 5) * 2.5 / 5
    return scaler * duration_ms / 1000


class AdvancedEventAnalyzer:
    """
    Analyzes raw tracking and detection data against the beatmap's ground
    truth to produce a detailed log of gameplay events, including slider
    breaks and spinner RPM.
    """

    def __init__(self, hit_objects, difficulty, timing_windows):
        """
        Initializes the analyzer with beatmap data.
        """
        self.hit_objects = hit_objects
        self.difficulty = difficulty
        self.timing_windows = timing_windows

        self.hit_objects_by_time = sorted(hit_objects, key=lambda x: x['time'])
        for obj in self.hit_objects_by_time:
            if obj.get('is_slider'):
                num_ticks = math.ceil(obj['slides'] * obj['length'] / 100)
                obj['slider_ticks_to_hit'] = list(range(1, num_ticks))
            if obj.get('is_spinner'):
                duration = obj['end_time'] - obj['time']
                obj['req_rotations'] = get_spinner_req_rotations(self.difficulty.get('OverallDifficulty', 5), duration)

        self.next_object_idx = 0

        self.active_slider = None
        self.active_spinner = None

        print_status("Advanced Event Analyzer initialized with beatmap data.")

    def analyze_frame(self, frame_num, video_info, cursor_pos):
        """
        Analyzes a single frame for game events.
        """
        current_time_ms = frame_num * 1000 / video_info['fps']
        events = []

        # --- 1. Check for misses ---
        if self.next_object_idx < len(self.hit_objects_by_time):
            next_obj = self.hit_objects_by_time[self.next_object_idx]
            if not next_obj.get('hit') and current_time_ms > next_obj['time'] + self.timing_windows['50']:
                events.append({'event': 'miss', 'obj': next_obj})
                next_obj['hit'] = True
                self.next_object_idx += 1

        # --- 2. Check for hits ---
        if cursor_pos and self.next_object_idx < len(self.hit_objects_by_time):
            obj = self.hit_objects_by_time[self.next_object_idx]
            time_diff = abs(current_time_ms - obj['time'])

            if not obj.get('hit') and time_diff <= self.timing_windows['50']:
                cs_radius_px = (109 - 9 * self.difficulty.get('CircleSize', 4)) / 2
                playfield_scale = cfg.PLAYFIELD_HEIGHT / 384.0
                scaled_radius = cs_radius_px * playfield_scale

                obj_screen_x = int(obj['x'] * (cfg.PLAYFIELD_WIDTH / 512.0) + cfg.PLAYFIELD_OFFSET_X)
                obj_screen_y = int(obj['y'] * (cfg.PLAYFIELD_HEIGHT / 384.0) + cfg.PLAYFIELD_OFFSET_Y)
                dist = math.hypot(cursor_pos[0] - obj_screen_x, cursor_pos[1] - obj_screen_y)

                if dist <= scaled_radius:
                    hit_type = '50'
                    if time_diff <= self.timing_windows['100']: hit_type = '100'
                    if time_diff <= self.timing_windows['300']: hit_type = '300'

                    events.append({'event': 'hit', 'type': hit_type, 'obj': obj})
                    obj['hit'] = True
                    # --- NEW: Record the exact time of the hit ---
                    obj['hit_time'] = current_time_ms

                    if obj.get('is_slider'):
                        self.active_slider = obj
                        self.active_slider['is_tracking'] = True
                    if obj.get('is_spinner'):
                        self.active_spinner = obj
                        self.active_spinner.update({
                            'total_rotation': 0, 'last_angle': None, 'rotations_done': 0
                        })

                    self.next_object_idx += 1

        # --- 3. Process active sliders ---
        if self.active_slider and cursor_pos:
            slider = self.active_slider
            slider_end_time = slider['time'] + slider.get('slider_duration', 0)

            if slider['is_tracking']:
                cs_radius_px = (109 - 9 * self.difficulty.get('CircleSize', 4)) / 2
                playfield_scale = cfg.PLAYFIELD_HEIGHT / 384.0
                follow_radius = cs_radius_px * playfield_scale * 2.4

                progress = (current_time_ms - slider['time']) / slider.get('slider_duration', 1)
                slide_progress = progress * slider['slides']
                current_slide = math.floor(slide_progress)
                progress_this_slide = slide_progress - current_slide
                if current_slide % 2 != 0: progress_this_slide = 1 - progress_this_slide

                if 0 <= progress_this_slide <= 1 and slider.get('curve'):
                    ball_pos = slider['curve'].evaluate(progress_this_slide)
                    ball_x = int(ball_pos[0, 0] * (cfg.PLAYFIELD_WIDTH / 512.0) + cfg.PLAYFIELD_OFFSET_X)
                    ball_y = int(ball_pos[1, 0] * (cfg.PLAYFIELD_HEIGHT / 384.0) + cfg.PLAYFIELD_OFFSET_Y)

                    dist_to_ball = math.hypot(cursor_pos[0] - ball_x, cursor_pos[1] - ball_y)
                    if dist_to_ball > follow_radius:
                        slider['is_tracking'] = False
                        events.append({'event': 'slider_break'})

            if slider['is_tracking'] and len(slider['slider_ticks_to_hit']) > 0:
                num_ticks_total = math.ceil(slider['slides'] * slider['length'] / 100)
                tick_to_check = slider['slider_ticks_to_hit'][0]
                tick_progress = tick_to_check / num_ticks_total
                tick_time = slider['time'] + (tick_progress / slider['slides']) * slider.get('slider_duration', 0)

                if current_time_ms >= tick_time:
                    events.append({'event': 'slider_tick'})
                    slider['slider_ticks_to_hit'].pop(0)

            if current_time_ms >= slider_end_time:
                self.active_slider = None

        # --- 4. Process active spinners ---
        if self.active_spinner and cursor_pos:
            spinner = self.active_spinner
            center_x, center_y = cfg.OUTPUT_RESOLUTION[0] // 2, cfg.OUTPUT_RESOLUTION[1] // 2

            dx = cursor_pos[0] - center_x
            dy = cursor_pos[1] - center_y
            current_angle = math.atan2(dy, dx)

            if spinner['last_angle'] is not None:
                delta_angle = current_angle - spinner['last_angle']
                if delta_angle > math.pi: delta_angle -= 2 * math.pi
                if delta_angle < -math.pi: delta_angle += 2 * math.pi

                spinner['total_rotation'] += delta_angle

                frame_duration_s = 1 / video_info['fps']
                rpm = abs(delta_angle / (2 * math.pi) / frame_duration_s * 60)
                events.append({'event': 'spinner_rpm_update', 'rpm': int(rpm)})

                if abs(spinner['total_rotation']) >= (spinner['rotations_done'] + 1) * 2 * math.pi:
                    spinner['rotations_done'] += 1
                    events.append({'event': 'spinner_rotation'})

            spinner['last_angle'] = current_angle

            if current_time_ms >= spinner['end_time']:
                if spinner['rotations_done'] >= spinner['req_rotations']:
                    events.append({'event': 'spinner_bonus'})
                self.active_spinner = None
                events.append({'event': 'spinner_rpm_update', 'rpm': 0})

        return events