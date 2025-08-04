# game_simulation.py
#
# A class to manage and calculate a simulated game state (HP, score, etc.)
# for the purpose of generating realistic training data.
# MODIFIED: HP logic has been removed, as it's now sourced from the replay file.
# MODIFIED: Now simulates slider tracking to determine when the follow circle should be visible.
# MODIFIED: Now stores the exact 'miss_time' on an object when it is missed.
# FIXED: Now correctly handles slider combo breaks.
# MODIFIED: Tracks last slider break time for Flashlight (FL) mod.
# FIXED: Corrected a boundary condition that caused a false slider break at the end of every slider.
# MODIFIED: Assumes perfect slider tracking when no cursor data (replay) is provided.
# FIXED: Now correctly accepts the 'no_misses' argument to prevent a TypeError.

import math
import config_generator as cfg


class GameSimulation:
    """
    Simulates the game state for rendering purposes, including slider follow tracking.
    """

    def __init__(self, difficulty, timing_windows, no_misses=False):
        """
        Initializes the simulation state.

        Args:
            difficulty (dict): The difficulty parameters of the beatmap.
            timing_windows (dict): The timing windows for 300/100/50 hits.
            no_misses (bool): If True, the simulation will not generate misses.
        """
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.accuracy = 100.0
        self.hits = {'300': 0, '100': 0, '50': 0, 'miss': 0}

        self.difficulty = difficulty
        self.timing_windows = timing_windows
        self.last_update_time_ms = 0
        self.processed_hit_objects = set()
        self.previous_key_state = {'m1': False, 'm2': False, 'k1': False, 'k2': False}
        self.m1_presses, self.m2_presses, self.k1_presses, self.k2_presses = 0, 0, 0, 0

        self.active_slider = None
        self.active_slider_is_tracking = True
        self.last_slider_break_time = 0.0
        self.no_misses = no_misses

    def _recalculate_accuracy(self):
        """Recalculates accuracy based on the current hit counts."""
        total_hits = sum(self.hits.values())
        if total_hits == 0: self.accuracy = 100.0; return
        weighted_hits = (self.hits['50'] * 50 + self.hits['100'] * 100 + self.hits['300'] * 300)
        self.accuracy = (weighted_hits / (total_hits * 300)) * 100.0 if total_hits > 0 else 100.0

    def update_state(self, current_time_ms, hit_objects, current_key_state, cursor_pos):
        """
        Updates the score, combo, and slider tracking state.

        Args:
            current_time_ms (float): The current timestamp in the simulation.
            hit_objects (list): The list of all hit objects in the beatmap.
            current_key_state (dict): The current state of m1, m2, k1, and k2 presses.
            cursor_pos (tuple): The (x, y) coordinates of the cursor from the replay.

        Returns:
            dict: A dictionary containing the current full game state.
        """
        # --- Handle Active Slider Logic ---
        is_slider_tracked = False
        if self.active_slider:
            slider_end_time = self.active_slider['time'] + self.active_slider.get('slider_duration', 0)

            if current_time_ms >= slider_end_time:
                self.active_slider = None
            # If no cursor data is available (i.e., no replay), assume perfect tracking.
            elif cursor_pos is None:
                is_slider_tracked = True
            else:
                slider = self.active_slider
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

                    cursor_screen_x = int(cursor_pos[0] * (cfg.PLAYFIELD_WIDTH / 512.0) + cfg.PLAYFIELD_OFFSET_X)
                    cursor_screen_y = int(cursor_pos[1] * (cfg.PLAYFIELD_HEIGHT / 384.0) + cfg.PLAYFIELD_OFFSET_Y)

                    dist_to_ball = math.hypot(cursor_screen_x - ball_x, cursor_screen_y - ball_y)
                    is_slider_tracked = dist_to_ball <= follow_radius

                    if self.active_slider_is_tracking and not is_slider_tracked:
                        self.combo = 0
                        self.active_slider_is_tracking = False
                        slider['break_time'] = current_time_ms
                        self.last_slider_break_time = current_time_ms

        # --- Score/Combo/Acc Updates from Hits/Misses ---
        for obj in hit_objects:
            obj_time = obj['time']
            if obj_time in self.processed_hit_objects:
                continue

            if 'hit_time' in obj and current_time_ms >= obj['hit_time']:
                self.processed_hit_objects.add(obj_time)
                hit_result = obj.get('hit_result', '300')
                if hit_result != 'miss':
                    self.hits[hit_result] += 1
                    self.combo += 1
                    if self.combo > self.max_combo: self.max_combo = self.combo
                    score_increase = {'300': 300, '100': 100, '50': 50}.get(hit_result, 0)
                    self.score += score_increase + (score_increase * (self.combo - 1) // 25)
                    if obj.get('is_slider'):
                        self.active_slider = obj
                        self.active_slider_is_tracking = True

            elif current_time_ms > obj_time + self.timing_windows['50'] and 'hit_time' not in obj:
                if not self.no_misses:
                    self.processed_hit_objects.add(obj_time)
                    self.hits['miss'] += 1
                    self.combo = 0
                    obj['miss_time'] = obj_time + self.timing_windows['50']

        # --- Key Press Counting ---
        k1_pressed = current_key_state['k1'] and not self.previous_key_state['k1']
        k2_pressed = current_key_state['k2'] and not self.previous_key_state['k2']
        m1_pressed = (current_key_state['m1'] and not self.previous_key_state['m1']) and not k1_pressed
        m2_pressed = (current_key_state['m2'] and not self.previous_key_state['m2']) and not k2_pressed
        if k1_pressed: self.k1_presses += 1
        if k2_pressed: self.k2_presses += 1
        if m1_pressed: self.m1_presses += 1
        if m2_pressed: self.m2_presses += 1
        self.previous_key_state = current_key_state.copy()

        self._recalculate_accuracy()
        self.last_update_time_ms = current_time_ms

        return {
            'score': self.score, 'combo': self.combo, 'accuracy': self.accuracy,
            'm1_presses': self.m1_presses, 'm2_presses': self.m2_presses,
            'k1_presses': self.k1_presses, 'k2_presses': self.k2_presses,
            'is_slider_tracked': is_slider_tracked,
            'last_slider_break_time': self.last_slider_break_time
        }