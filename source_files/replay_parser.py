# replay_parser.py
#
# Contains the definitive logic for parsing raw osu! replay data (.osr) into an
# absolute timeline. This logic is a direct and rigorous translation of the
# timeline processing found in the circleguard v5.4.3 project. It correctly
# handles all known replay format edge cases, including intro skips.

import numpy as np
from osrparse import ReplayEventOsu


def process_replay_to_numpy_arrays(raw_replay_data):
    """
    Converts raw replay event data with time deltas into a tuple of synchronized
    NumPy arrays (t, xy, k).

    This is a direct translation of the logic in `circleguard.loadables.Replay._process_replay_data`.

    Args:
        raw_replay_data (list): The list of ReplayEvent objects from osrparse.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
            absolute time array (t), the position array (xy), and the key
            state array (k). Returns (None, None, None) if data is invalid.
    """
    if not raw_replay_data:
        return None, None, None

    # Per osu! specification, the final event is a random number seed (-12345) and must be ignored.
    if raw_replay_data[-1].time_delta == -12345:
        gameplay_events = raw_replay_data[:-1]
    else:
        gameplay_events = raw_replay_data

    if not gameplay_events or not isinstance(gameplay_events[0], ReplayEventOsu):
        return None, None, None

    # Ignore the first frame if its time_delta is 0, which is an invalid frame.
    if gameplay_events[0].time_delta == 0:
        gameplay_events = gameplay_events[1:]
        if not gameplay_events:
            return None, None, None

    # This logic is a direct port from circleguard.
    data = [[], [], [], []]  # t, x, y, k
    events_iterator = iter(gameplay_events)

    try:
        # The first event's delta is treated specially. It initializes the clock.
        first_event = next(events_iterator)
    except StopIteration:
        return None, None, None

    running_t = float(first_event.time_delta)

    # CRITICAL FIX: In circleguard's logic, the first frame is processed at the end,
    # before the final sort. It is NOT added to the data list immediately.
    # This is the key to correctly handling skipped replays.
    initial_frame_data = (running_t, first_event.x, first_event.y, int(first_event.keys))

    highest_running_t = running_t
    last_positive_frame = first_event
    last_positive_frame_cum_time = running_t
    previous_frame = first_event

    for event in events_iterator:
        was_in_negative_section = running_t < highest_running_t

        delta_t = float(event.time_delta)
        running_t += delta_t
        highest_running_t = max(highest_running_t, running_t)

        if running_t < highest_running_t:
            if not was_in_negative_section:
                last_positive_frame = previous_frame
                last_positive_frame_cum_time = running_t - delta_t
            previous_frame = event
            continue

        if was_in_negative_section:
            interp_time = last_positive_frame_cum_time
            prev_event_time = running_t - delta_t
            curr_event_time = running_t

            if curr_event_time - prev_event_time == 0:
                interp_x, interp_y = event.x, event.y
            else:
                progress = (interp_time - prev_event_time) / (curr_event_time - prev_event_time)
                interp_x = previous_frame.x + (event.x - previous_frame.x) * progress
                interp_y = previous_frame.y + (event.y - previous_frame.y) * progress

            data[0].append(interp_time)
            data[1].append(interp_x)
            data[2].append(interp_y)
            data[3].append(int(last_positive_frame.keys))

        data[0].append(running_t)
        data[1].append(event.x)
        data[2].append(event.y)
        data[3].append(int(event.keys))
        previous_frame = event

    # Add the initial frame data now, before the sort.
    data[0].append(initial_frame_data[0])
    data[1].append(initial_frame_data[1])
    data[2].append(initial_frame_data[2])
    data[3].append(initial_frame_data[3])

    block = np.array(data)

    t = np.array(block[0], dtype=float)
    xy = np.array([block[1], block[2]], dtype=float).T
    k = np.array(block[3], dtype=int)

    # A final sort by time is critical to place any interpolated frames and the initial frame correctly.
    t_sort = np.argsort(t, kind="stable")
    t = t[t_sort]
    xy = xy[t_sort]
    k = k[t_sort]

    return t, xy, k