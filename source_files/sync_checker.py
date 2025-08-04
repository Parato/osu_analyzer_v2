# sync_checker.py
# A diagnostic tool to inspect and compare the processed timelines of a beatmap and a replay.

import argparse
from osrparse import Replay

# We must import the exact parser function our pipeline uses to ensure we are testing the real code.
from osu_parser import parse_beatmap


def run_sync_check(beatmap_path, replay_path):
    """
    Loads a beatmap and replay, processes their timelines, and prints them for comparison.
    """
    print("=" * 60)
    print(f"Running Sync Check for:")
    print(f"  Beatmap: {beatmap_path}")
    print(f"  Replay:  {replay_path}")
    print("=" * 60)

    # --- 1. Process the Beatmap Timeline ---
    # We use the exact same function as the main pipeline.
    try:
        hit_objects, _, timing_points, _, _ = parse_beatmap(beatmap_path, apply_dt=False, apply_hr=False)
        if not hit_objects:
            print("\n[ERROR] Failed to parse beatmap or no hit objects found.")
            return
    except Exception as e:
        print(f"\n[ERROR] An exception occurred while parsing the beatmap: {e}")
        return

    print("\n--- BEATMAP TIMELINE (Processed by osu_parser.py) ---")
    print("This is the timeline the renderer uses for placing objects.")

    first_tp_time = timing_points[0]['time'] if timing_points else "N/A"
    print(f"\nFirst Timing Point (Offset used): {first_tp_time}ms")

    print("\nFirst 10 Hit Objects (Final Timestamps):")
    for i, obj in enumerate(hit_objects[:10]):
        obj_type = "Circle"
        if obj.get('is_slider'): obj_type = "Slider"
        if obj.get('is_spinner'): obj_type = "Spinner"
        print(f"  - Obj {i:02d} ({obj_type:<7}): {obj['time']:>7}ms")

    # --- 2. Process the Replay Timeline ---
    # We use the exact same logic as the main pipeline.
    try:
        replay = Replay.from_path(replay_path)

        gameplay_events = replay.replay_data[:-1]  # Exclude final RNG seed

        current_time = 0.0
        for event in gameplay_events:
            current_time += event.time_delta
            event.time_ms = current_time

    except Exception as e:
        print(f"\n[ERROR] An exception occurred while parsing the replay: {e}")
        return

    print("\n\n--- REPLAY TIMELINE (Processed by autogen_dataset.py) ---")
    print("This is the timeline used for cursor position and key presses.")
    print("\nFirst 40 Replay Events (Final Timestamps):")
    for i, event in enumerate(gameplay_events[:40]):
        print(f"  - Event {i:02d}: Delta={event.time_delta:<7}ms -> Absolute Time={event.time_ms:>8.0f}ms")

    print("\n" + "=" * 60)
    print("Sync Check Complete.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Osu! Beatmap-Replay Sync Checker")
    parser.add_argument("beatmap_path", help="Path to the .osu beatmap file.")
    parser.add_argument("replay_path", help="Path to the .osr replay file.")
    args = parser.parse_args()

    run_sync_check(args.beatmap_path, args.replay_path)