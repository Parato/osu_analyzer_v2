import numpy as np
from typing import List, Dict


class DifficultyCalculator:
    """
    Calculates a simplified star rating for a map based on Aim and Speed strains.
    This version uses more advanced techniques like normalization and value squashing
    to produce more realistic and stable star ratings.
    """
    # Constants tuned for the new, more controlled calculation
    DECAY_BASE = 0.3  # How much strain is lost per second
    AIM_DIFFICULTY_MULTIPLIER = 22
    SPEED_DIFFICULTY_MULTIPLIER = 18
    STAR_RATING_MULTIPLIER = 0.04  # Final scaling factor

    def _preprocess_objects(self, hit_objects: List[Dict]) -> List[Dict]:
        """Groups circles that appear at almost the same time (stacks) into a single event."""
        if not hit_objects:
            return []

        sorted_objects = sorted(hit_objects, key=lambda x: x['start_ts'])
        processed_list = []
        i = 0
        while i < len(sorted_objects):
            current_group = [sorted_objects[i]]
            j = i + 1
            while j < len(sorted_objects) and (sorted_objects[j]['start_ts'] - sorted_objects[i]['start_ts']) < 0.01:
                current_group.append(sorted_objects[j])
                j += 1

            avg_x = sum(c['x'] for c in current_group) / len(current_group)
            avg_y = sum(c['y'] for c in current_group) / len(current_group)

            processed_list.append({
                'start_ts': sorted_objects[i]['start_ts'],
                'x': avg_x,
                'y': avg_y,
            })
            i = j
        return processed_list

    def calculate_star_rating(self, hit_objects: List[Dict], circle_radius: float) -> float:
        """Calculates the final star rating for the given list of hit objects."""
        if len(hit_objects) < 2 or circle_radius <= 0:
            return 0.0

        processed_objects = self._preprocess_objects(hit_objects)
        if len(processed_objects) < 2:
            return 0.0

        aim_strains = []
        speed_strains = []

        current_aim_strain = 0
        current_speed_strain = 0

        for i in range(1, len(processed_objects)):
            prev_obj = processed_objects[i - 1]
            curr_obj = processed_objects[i]

            # Use a minimum time delta to prevent division by zero or infinity
            time_delta = max(curr_obj['start_ts'] - prev_obj['start_ts'], 0.02)

            # --- AIM STRAIN CALCULATION ---
            distance = np.linalg.norm([curr_obj['x'] - prev_obj['x'], curr_obj['y'] - prev_obj['y']])
            # Normalize distance by circle diameter.
            normalized_distance = distance / (2 * circle_radius)
            # Use tanh to squash the aim value, preventing explosive growth.
            aim_value = self.AIM_DIFFICULTY_MULTIPLIER * np.tanh(normalized_distance / 2.0)

            # --- SPEED STRAIN CALCULATION ---
            # Exponential decay makes the value high for small time_delta, and low for high time_delta.
            speed_value = self.SPEED_DIFFICULTY_MULTIPLIER * np.exp(-time_delta * 4)

            # Apply decay to the current strain values based on how long ago the last object was.
            decay_factor = self.DECAY_BASE ** time_delta
            current_aim_strain *= decay_factor
            current_speed_strain *= decay_factor

            # Add the new strain values
            current_aim_strain += aim_value
            current_speed_strain += speed_value

            aim_strains.append(current_aim_strain)
            speed_strains.append(current_speed_strain)

        aim_difficulty = self._calculate_difficulty_value(aim_strains)
        speed_difficulty = self._calculate_difficulty_value(speed_strains)

        # The final star rating is a combination of the aim and speed difficulties.
        star_rating = (aim_difficulty + speed_difficulty) * self.STAR_RATING_MULTIPLIER
        return star_rating

    def _calculate_difficulty_value(self, strains: List[float]) -> float:
        """Calculates a final difficulty value from a list of strains by taking a weighted average of the peaks."""
        if not strains:
            return 0.0

        strains.sort(reverse=True)

        difficulty = 0.0
        weight = 0.95
        total_weight = 0.0

        for strain in strains:
            difficulty += strain * weight
            total_weight += weight
            weight *= 0.95

        return difficulty / total_weight if total_weight > 0 else 0.0
