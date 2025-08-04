# visualizer.py
#
# Contains logic for creating a visualization of the detection and tracking process.
# It draws bounding boxes, class names, and track IDs onto video frames.
# MODIFIED: Now correctly draws only the single, highest-confidence cursor.
# MODIFIED: Labels for tracked objects now include the confidence score.
# MODIFIED: Updated class colors to align with the simplified training data.
# MODIFIED: Removed the game state HUD, as it can no longer be calculated without a beatmap.

import cv2

# --- Color Configuration (BGR format) ---
CLASS_COLORS = {
    "hit_circle": (255, 153, 51),      # Light Blue
    "cursor": (0, 0, 255),             # Red
    "spinner": (255, 102, 255),        # Pink
    "hit_miss": (0, 255, 255),         # Yellow (for misses)
    "default": (0, 255, 0)             # Green
}


class Visualizer:
    """Handles drawing visualizations and writing the output video."""

    def __init__(self, output_path, video_info):
        """
        Initializes the visualizer and the video writer.

        Args:
            output_path (str): Path to save the output video file.
            video_info (dict): Dictionary containing properties of the source video.
        """
        self.writer = None
        self.video_width = video_info.get('width', 1280)
        self.video_height = video_info.get('height', 720)

        if not output_path:
            return

        # NOTE: The following line is correct. Some IDEs or linters may show a warning like
        # "Cannot find reference 'VideoWriter_fourcc'". This is a known issue with the
        # cv2 library's type hints and can be safely ignored.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        fps = video_info.get('fps', 30)

        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (self.video_width, self.video_height))
        if self.writer.isOpened():
            print(f"[INFO] Output video will be saved to: {output_path}")
        else:
            print(f"[WARN] Could not open video writer for path: {output_path}. No video will be saved.")
            self.writer = None

    def draw_frame(self, frame, detections, active_tracks, cursor_detection=None):
        """
        Draws all detections and tracking information onto a single frame.

        Args:
            frame (numpy.ndarray): The video frame to draw on.
            detections (list): The list of raw detections for this frame.
            active_tracks (dict): The dictionary of active tracks from the tracker.
            cursor_detection (dict, optional): The single best cursor detection for this frame.
        """
        # Draw all non-cursor active tracks.
        for track_id, track in active_tracks.items():
            if track['class'] == 'cursor':
                continue

            x, y, w, h = track['box']
            class_name = track['class']
            color = CLASS_COLORS.get(class_name, CLASS_COLORS['default'])

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Display the track ID, class, and confidence for all non-cursor objects
            confidence = track.get('confidence', 0.0)
            label = f"ID: {track_id} ({class_name}) ({confidence:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=color, thickness=2)

        # Draw the single best cursor detection, if it exists
        if cursor_detection:
            x, y, w, h = cursor_detection['box']
            class_name = cursor_detection['class']
            confidence = cursor_detection['confidence']
            color = CLASS_COLORS.get(class_name, CLASS_COLORS['default'])

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # For the main cursor, show its confidence instead of a track ID
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=color, thickness=2)

        return frame

    def write(self, frame):
        """Writes a frame to the output video file."""
        if self.writer:
            self.writer.write(frame)

    def release(self):
        """Releases the video writer resource."""
        if self.writer:
            self.writer.release()
            print("[INFO] Output video has been saved.")