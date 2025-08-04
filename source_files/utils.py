# utils.py
#
# Contains utility functions shared across different modules.

def print_status(message, level="INFO"):
    """Prints a formatted status message to the console."""
    print(f"[{level}] {message}")

# --- NEW: Centralized function for finding the best cursor detection ---
def get_best_cursor_detections(all_detections_by_frame):
    """
    Finds the single best cursor detection for each frame based on confidence.
    This is used to ensure the cursor is always visualized if detected, and provides
    a stable cursor position for analysis.

    Args:
        all_detections_by_frame (dict): A dictionary where keys are frame numbers
                                        and values are lists of detection dicts.

    Returns:
        dict: A dictionary where keys are frame numbers and values are the single
              best cursor detection object for that frame.
    """
    cursor_detections_by_frame = {}
    for frame_num, detections in all_detections_by_frame.items():
        cursor_detections = [d for d in detections if d['class'] == 'cursor']
        if cursor_detections:
            # Find the cursor detection with the highest confidence score
            best_cursor_det = max(cursor_detections, key=lambda x: x['confidence'])
            cursor_detections_by_frame[frame_num] = best_cursor_det
    return cursor_detections_by_frame