# follow_points.py
#
# Contains the logic for rendering the animated "follow points" that connect
# consecutive objects in a combo.
# FIXED: The code now correctly handles the return type from get_animated_frame,
#        preventing crashes when trying to access Image attributes on a list.
# FIXED: Opacity is now correctly applied to a copy of the asset, preventing
#        modification of the original cached asset.

from PIL import Image
import math
import config_generator as cfg

def get_animated_frame(asset, current_time_ms, anim_start_time, framerate):
    """
    Gets the correct frame for an animated asset.
    """
    if not isinstance(asset, list) or not asset:
        return asset  # Return single image or None

    # Default to 60fps if animationframerate is not set or invalid
    if framerate <= 0:
        framerate = 60

    elapsed_time = current_time_ms - anim_start_time
    # Ensure we don't divide by zero
    frame_duration_ms = 1000 / framerate
    if frame_duration_ms == 0:
        return asset[0]

    frame_index = int(elapsed_time / frame_duration_ms)

    # Loop the animation
    if len(asset) > 0:
        return asset[frame_index % len(asset)]
    return None


def render_follow_points(frame_image, assets, prev_obj, current_obj, current_time_ms, ar_ms):
    """
    Renders the animated follow points between two hit objects.

    Args:
        frame_image (PIL.Image): The main canvas to draw on.
        assets (dict): The loaded skin assets.
        prev_obj (dict): The previous hit object in the combo.
        current_obj (dict): The current hit object to draw points towards.
        current_time_ms (float): The current timestamp in the simulation.
        ar_ms (float): The approach rate in milliseconds.
    """
    follow_point_frames = assets.get('followpoint')
    # Skin must have follow points, and they must be a list of frames.
    if not follow_point_frames or not isinstance(follow_point_frames, list):
        return

    # Determine the visibility window for the follow points
    fade_in_start_time = prev_obj['time'] - ar_ms
    fade_in_end_time = current_obj['time'] - ar_ms

    if not (fade_in_start_time < current_time_ms < current_obj['time']):
        return # Not in the visible time window

    # Calculate opacity based on the fade-in period
    if current_time_ms < fade_in_end_time:
        time_into_fade = current_time_ms - fade_in_start_time
        duration = fade_in_end_time - fade_in_start_time
        opacity = time_into_fade / duration if duration > 0 else 1.0
    else:
        opacity = 1.0 # Fully visible until the object is hit

    opacity = max(0.0, min(1.0, opacity))
    if opacity <= 0:
        return

    # Get the specific animated frame for the current time
    framerate = int(assets.get('ini', {}).get('General', {}).get('animationframerate', -1))
    point_frame = get_animated_frame(follow_point_frames, current_time_ms, 0, framerate)

    # --- FIX: Check that we have a valid Image object before proceeding ---
    if not isinstance(point_frame, Image.Image):
        return

    # Get screen coordinates for the two objects
    prev_x = int(prev_obj.get('render_x', prev_obj['x']) * (cfg.PLAYFIELD_WIDTH / 512.0) + cfg.PLAYFIELD_OFFSET_X)
    prev_y = int(prev_obj.get('render_y', prev_obj['y']) * (cfg.PLAYFIELD_HEIGHT / 384.0) + cfg.PLAYFIELD_OFFSET_Y)
    curr_x = int(current_obj.get('render_x', current_obj['x']) * (cfg.PLAYFIELD_WIDTH / 512.0) + cfg.PLAYFIELD_OFFSET_X)
    curr_y = int(current_obj.get('render_y', current_obj['y']) * (cfg.PLAYFIELD_HEIGHT / 384.0) + cfg.PLAYFIELD_OFFSET_Y)

    dist = math.hypot(curr_x - prev_x, curr_y - prev_y)
    angle = math.atan2(curr_y - prev_y, curr_x - prev_x)

    # --- FIX: Spacing is now accessed safely ---
    spacing = point_frame.width * 1.5
    if spacing <= 0: return

    num_points = int(dist / spacing)

    # --- FIX: Apply opacity to a *copy* of the frame to avoid modifying the asset cache ---
    if opacity < 1.0:
        alpha = point_frame.getchannel('A')
        frame_to_paste = point_frame.copy()
        frame_to_paste.putalpha(alpha.point(lambda p: int(p * opacity)))
    else:
        frame_to_paste = point_frame

    # Draw the points along the line connecting the objects
    for i in range(1, num_points):
        d = i * spacing
        x = int(prev_x + d * math.cos(angle))
        y = int(prev_y + d * math.sin(angle))

        # --- FIX: width/height accessed safely, and frame_image.paste is now guaranteed to work ---
        top_left_x = x - frame_to_paste.width // 2
        top_left_y = y - frame_to_paste.height // 2

        frame_image.paste(frame_to_paste, (top_left_x, top_left_y), frame_to_paste)