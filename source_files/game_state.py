# game_state.py
#
# MODIFIED: This class has been rewritten to function as a state and statistics
# accumulator for the video analysis pipeline. It no longer handles real-time
# Pygame input and instead tracks score, combo, accuracy, and other stats
# based on events from the AdvancedEventAnalyzer.

class GameState:
    """
    Manages and calculates game state and statistics based on a stream of
    analyzed gameplay events.
    """
    def __init__(self):
        """
        Initializes the game state tracker. No arguments are needed as it
        is not tied to a real-time configuration.
        """
        self.score = 0
        self.combo = 0
        self.rpm = 0
        self.hits = {'300': 0, '100': 0, '50': 0}
        self.misses = 0
        self.slider_ticks = 0
        self.slider_breaks = 0
        self.accuracy = 100.0

    def _update_accuracy(self):
        """
        Recalculates the accuracy based on the current hit counts.
        Accuracy = ( (300 * H300) + (100 * H100) + (50 * H50) ) / ( (TotalHits) * 300 )
        """
        total_possible_points = (self.hits['300'] + self.hits['100'] + self.hits['50'] + self.misses) * 300
        if total_possible_points == 0:
            self.accuracy = 100.0
            return

        actual_points = (self.hits['300'] * 300) + (self.hits['100'] * 100) + (self.hits['50'] * 50)
        self.accuracy = (actual_points / total_possible_points) * 100.0

    def add_hit(self, hit_type):
        """
        Processes a successful hit on a circle or slider head.
        Args:
            hit_type (str): The type of hit ('300', '100', or '50').
        """
        if hit_type in self.hits:
            self.hits[hit_type] += 1
            self.score += int(hit_type)
            self.combo += 1
            self._update_accuracy()

    def add_miss(self):
        """Processes a missed circle or slider head."""
        self.misses += 1
        self.combo = 0
        self._update_accuracy()

    def add_slider_tick(self):
        """Processes a successful hit on a slider tick."""
        self.score += 10
        self.combo += 1

    def add_slider_break(self):
        """Processes a slider break (losing combo)."""
        self.combo = 0
        self.slider_breaks += 1

    def set_rpm(self, rpm):
        """
        Updates the current spinner RPM.
        Args:
            rpm (int): The new RPM value.
        """
        self.rpm = rpm

    def add_spinner_rotation_points(self):
        """Adds points for a full rotation of a spinner."""
        self.score += 100

    def add_spinner_bonus_points(self):
        """Adds bonus points for clearing a spinner."""
        self.score += 1000

    def get_current_state(self):
        """
        Returns a dictionary of the current game state for visualization.
        """
        return {
            'score': self.score,
            'combo': self.combo,
            'accuracy': self.accuracy,
            'rpm': self.rpm
        }