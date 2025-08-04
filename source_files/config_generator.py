# config_generator.py
#
# Stores configuration parameters for the automated dataset generator.
# MODIFIED: Removed 'spinner' from the class map and re-indexed subsequent classes.

# --- Output Configuration ---
OUTPUT_RESOLUTION = (640, 480)
FRAME_RATE = 60

# --- Playfield Configuration ---
# The playfield is scaled to fit inside the output resolution, with padding
# to ensure the largest circles (CS0) are never rendered off-screen.
_CS0_RADIUS_OSU_PX = 54.5
_PLAYFIELD_SCALE = OUTPUT_RESOLUTION[1] / 384.0  # Scale based on height
PLAYFIELD_PADDING = int(_CS0_RADIUS_OSU_PX * _PLAYFIELD_SCALE)
PLAYFIELD_WIDTH = OUTPUT_RESOLUTION[0] - (2 * PLAYFIELD_PADDING)
PLAYFIELD_HEIGHT = OUTPUT_RESOLUTION[1] - (2 * PLAYFIELD_PADDING)
PLAYFIELD_OFFSET_X = PLAYFIELD_PADDING
PLAYFIELD_OFFSET_Y = PLAYFIELD_PADDING

# --- Annotation Configuration ---
# Class IDs for annotations. MUST match the training YAML file.
# UPDATED: The class map has been simplified. 'spinner' and 'hit_success' were removed.
CLASS_MAP = {
    "hit_circle": 0,
    "cursor": 1,
    "hit_miss": 2
}

# --- Animation & Timing Configuration ---
# General
FADE_OUT_DURATION_MS = 150  # How long objects take to fade out after being hit/completed

# Hidden (HD) Mod
HD_FADE_IN_DURATION_MULTIPLIER = 0.4  # Circle fades in during the first 40% of AR time
HD_SLIDER_BODY_FADE_OUT_DURATION_MS = 240  # How long the slider body takes to fade out AFTER being hit

# Judgement Animations
HIT_BURST_SCALE_START = 1.0  # The scale of the hit burst at the moment of the hit
HIT_BURST_SCALE_END = 1.4  # The final scale of the hit burst at the end of its animation
HIT_BURST_DURATION_MS = 300  # How long the hit burst animation lasts
MISS_FADE_IN_DURATION_MS = 40  # How long the miss symbol takes to fade in
MISS_FADE_OUT_DURATION_MS = 260  # How long the miss symbol takes to fade out after fading in
MISS_ROTATION_DEGREES = 20 # The total amount the miss sprite will rotate
MISS_GRAVITY_DISTANCE_PX = 25 # How far the miss sprite will fall

# --- UI Configuration ---
UI_SCALE = 0.75  # The master scale for all pre-rendered UI assets (score, combo, etc.)
UI_HP_BAR_MAX_WIDTH_PX = 250  # A hard pixel cap on the HP bar's width.
UI_HP_BAR_MIN_WIDTH_PX = 240  # A hard pixel cap on the HP bar's minimum width.
UI_OVERLAY_HEIGHT_RATIO = 0.35  # Final overlay height as a percentage of screen height.