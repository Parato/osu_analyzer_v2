# tracker.py
#
# Contains the logic for tracking detected objects over multiple frames.
# UPDATED: The set of trackable classes has been simplified.
# UPDATED: Class compatibility for 'slider_head' is no longer needed.
# MODIFIED: Tracks now store the confidence score of their last associated detection.
# MODIFIED: Added 'hit_miss' to the set of trackable classes.
# MODIFIED: Implemented a two-stage matching cascade in the update method to prioritize
#           high-confidence matches and improve tracking stability.

import math
import config


def get_center(box):
    """Calculates the center of a bounding box."""
    x, y, w, h = box
    return x + w // 2, y + h // 2


class ObjectTracker:
    """
    Tracks objects by associating new detections with existing tracks
    based on proximity and class compatibility.
    """

    def __init__(self, max_age=None, dist_thresh=None):
        """
        Initializes the tracker.

        Args:
            max_age (int, optional): Override the default max track age.
            dist_thresh (int, optional): Override the default distance threshold.
        """
        self.active_tracks = {}
        self.finished_tracks = []
        self.next_track_id = 0
        self.trackable_classes = {
            "hit_circle", "cursor", "spinner", "hit_miss"
        }
        # MODIFIED: Compatibility is no longer needed as slider_head is now hit_circle
        self.compatible_classes = []

        self.max_age = max_age if max_age is not None else config.MAX_TRACK_AGE
        self.dist_thresh = dist_thresh if dist_thresh is not None else config.TRACKING_DIST_THRESHOLD

        # --- NEW: Thresholds for the matching cascade ---
        # Stricter distance for high-confidence matches
        self.high_conf_dist_thresh = self.dist_thresh * 0.75
        # Confidence score required to be considered in the first pass
        self.high_conf_score_thresh = 0.80

        print("[INFO] Initializing simple distance-based tracker with:")
        print(f"[INFO]   - Max Track Age: {self.max_age} frames")
        print(f"[INFO]   - Distance Threshold: {self.dist_thresh} pixels")
        print(f"[INFO]   - Matching Cascade: ENABLED")

    def _are_classes_compatible(self, class1, class2):
        """Checks if two classes are considered compatible for tracking."""
        if class1 == class2:
            return True
        for group in self.compatible_classes:
            if class1 in group and class2 in group:
                return True
        return False

    def update(self, detections, frame_number):
        """
        Updates the tracker using a two-stage matching cascade.

        Stage 1: Match tracks to high-confidence detections using a strict distance threshold.
        Stage 2: Match remaining tracks to any remaining detections using the standard threshold.
        """
        # 1. Increment the age of all active tracks first.
        for track_id in self.active_tracks:
            self.active_tracks[track_id]['age'] += 1

        unmatched_track_ids = set(self.active_tracks.keys())
        unmatched_detection_indices = set(range(len(detections)))
        matches = []

        # --- CASCADE STAGE 1: HIGH-CONFIDENCE MATCHING ---
        # Create a list of possible high-confidence matches
        possible_high_conf_matches = []
        for track_id in unmatched_track_ids:
            track = self.active_tracks[track_id]
            for det_idx in unmatched_detection_indices:
                detection = detections[det_idx]
                # Check for high confidence and class compatibility
                if detection['confidence'] >= self.high_conf_score_thresh and \
                        self._are_classes_compatible(track['class'], detection['class']):

                    track_center = get_center(track['box'])
                    det_center = get_center(detection['box'])
                    dist = math.hypot(det_center[0] - track_center[0], det_center[1] - track_center[1])

                    # Use a stricter distance threshold for this pass
                    if dist < self.high_conf_dist_thresh:
                        possible_high_conf_matches.append((dist, track_id, det_idx))

        # Greedily assign the best high-confidence matches
        possible_high_conf_matches.sort(key=lambda x: x[0])
        for dist, track_id, det_idx in possible_high_conf_matches:
            if track_id in unmatched_track_ids and det_idx in unmatched_detection_indices:
                matches.append((track_id, det_idx))
                unmatched_track_ids.remove(track_id)
                unmatched_detection_indices.remove(det_idx)

        # --- CASCADE STAGE 2: STANDARD MATCHING ---
        # Create a list of possible matches for the remaining items
        possible_standard_matches = []
        for track_id in unmatched_track_ids:
            track = self.active_tracks[track_id]
            for det_idx in unmatched_detection_indices:
                detection = detections[det_idx]
                if self._are_classes_compatible(track['class'], detection['class']):
                    track_center = get_center(track['box'])
                    det_center = get_center(detection['box'])
                    dist = math.hypot(det_center[0] - track_center[0], det_center[1] - track_center[1])

                    # Use the standard, more lenient distance threshold
                    if dist < self.dist_thresh:
                        possible_standard_matches.append((dist, track_id, det_idx))

        # Greedily assign the best remaining matches
        possible_standard_matches.sort(key=lambda x: x[0])
        for dist, track_id, det_idx in possible_standard_matches:
            if track_id in unmatched_track_ids and det_idx in unmatched_detection_indices:
                matches.append((track_id, det_idx))
                unmatched_track_ids.remove(track_id)
                unmatched_detection_indices.remove(det_idx)

        # 4. Update all matched tracks based on the two cascade passes
        for track_id, det_idx in matches:
            track = self.active_tracks[track_id]
            detection = detections[det_idx]
            track['box'] = detection['box']
            track['class'] = detection['class']  # Class might change (e.g., circle to slider_head)
            track['confidence'] = detection['confidence']
            track['age'] = 0
            track['last_seen_frame'] = frame_number

        # 5. Create new tracks for any detections that were not matched.
        for det_idx in unmatched_detection_indices:
            det = detections[det_idx]
            if det['class'] in self.trackable_classes:
                self.active_tracks[self.next_track_id] = {
                    'id': self.next_track_id,
                    'class': det['class'],
                    'box': det['box'],
                    'confidence': det['confidence'],
                    'spawn_frame': frame_number,
                    'last_seen_frame': frame_number,
                    'age': 0
                }
                self.next_track_id += 1

        # 6. Clean up old tracks that have become too old.
        for track_id in list(self.active_tracks.keys()):
            if self.active_tracks[track_id]['age'] > self.max_age:
                track = self.active_tracks[track_id]
                finished_event = {
                    'id': track['id'],
                    'class': track['class'],
                    'spawn_frame': track['spawn_frame'],
                    'despawn_frame': track['last_seen_frame'],
                    'box': track['box']
                }
                self.finished_tracks.append(finished_event)
                del self.active_tracks[track_id]

    def finalize(self, last_frame_number):
        """Moves any remaining active tracks to the finished list."""
        for track_id, track in self.active_tracks.items():
            self.finished_tracks.append({
                'id': track['id'],
                'class': track['class'],
                'spawn_frame': track['spawn_frame'],
                'despawn_frame': last_frame_number,
                'box': track['box']
            })
        self.active_tracks.clear()
        return self.finished_tracks