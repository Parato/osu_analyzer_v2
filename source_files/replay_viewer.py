# replay_viewer.py

import argparse
from osrparse import Replay

# --- Configuration ---
# Set this to how many replay events you want to see.
# Set to -1 to see all of them (can be very long).
MAX_EVENTS_TO_SHOW = -1


def main():
    parser = argparse.ArgumentParser(description="A simple tool to view the contents of an .osr replay file.")
    parser.add_argument("replay_path", help="The full path to the .osr replay file.")
    args = parser.parse_args()

    try:
        # Load the replay file using the osrparse library
        replay = Replay.from_path(args.replay_path)
    except Exception as e:
        print(f"Error: Could not load or parse the replay file at {args.replay_path}")
        print(f"Reason: {e}")
        return

    print("=" * 40)
    print("      OSU! REPLAY FILE INSPECTOR")
    print("=" * 40)

    # --- Print General Replay Information ---
    print("\n[--- General Info ---]")
    print(f"Beatmap Hash: {replay.beatmap_hash}")
    print(f"Player Name:  {replay.username}")
    print(f"Replay Hash:  {replay.replay_hash}")
    print(f"Mods:         {replay.mods.name}")
    print(f"Count 300s:   {replay.count_300}")
    print(f"Count 100s:   {replay.count_100}")
    print(f"Count 50s:    {replay.count_50}")
    print(f"Count Misses: {replay.count_miss}")
    print(f"Count Gekis:  {replay.count_geki}")
    print(f"Count Katus:  {replay.count_katu}")
    print(f"Max Combo:    {replay.max_combo}")
    # --- FIXED LINE ---
    # The correct attribute is 'perfect', not 'is_perfect'.
    print(f"Is Perfect:   {replay.perfect}")
    print(f"Timestamp:    {replay.timestamp}")

    # --- Print Replay Data Events ---
    print("\n[--- Replay Events (Cursor and Key Presses) ---]")
    print(f"(Showing first {MAX_EVENTS_TO_SHOW} events if available...)")
    print("-" * 40)

    # The replay_data list contains all the core events
    events_to_show = replay.replay_data
    if MAX_EVENTS_TO_SHOW != -1:
        events_to_show = replay.replay_data[:MAX_EVENTS_TO_SHOW]

    for i, event in enumerate(events_to_show):
        # time_delta is the time since the last event
        # x and y are cursor coordinates
        # keys is a bitmask of pressed keys (e.g., M1, M2, K1, K2)
        print(
            f"Event {i:04d}: time_delta={event.time_delta:<5}ms | X={event.x:<5.1f} | Y={event.y:<5.1f} | Keys={event.keys}")

    # --- Print Life Bar Data ---
    print("\n[--- Life Bar Graph ---]")
    if replay.life_bar_graph:
        for event in replay.life_bar_graph:
            # time is the absolute timestamp
            # life is the HP value (0.0 to 1.0)
            print(f"Time: {event.time:<7}ms | HP: {event.life:.2f}")
    else:
        print("No life bar data found in this replay.")

    print("\n" + "=" * 40)
    print("End of Replay Data")
    print("=" * 40)


if __name__ == "__main__":
    main()