import numpy as np
import pynapple as nap

EVENT_COL = ["EventGo_cue", "EventTarget_Onset"]
BEHAV_INFO_COL = ["target_ID", "target_dir", "result"]
BEHAV_VAR_COL = [
    "cursor_pos_x",
    "cursor_pos_y",
    "cursor_vel_x",
    "cursor_vel_y",
    "cursor_acc_x",
    "cursor_acc_y",
]


def get_event_timestamps(behav, info, event_name):
    """
    Get the timestamps of a specific event from the behavioral data.

    Parameters:
    - behav: The behavioral data (nap.TsdFrame).
    - event_name: The name of the event to extract timestamps for.

    Returns:
    - A nap.Ts object containing the timestamps of the specified event.
    """
    t = behav.times()

    # get a binary mask of event timestamps
    bw = behav[event_name] == True

    # get info about the event
    event_info = info.as_dataframe().iloc[np.where(bw)[0]].reset_index(drop=True)
    behav_info = behav.as_dataframe().iloc[np.where(bw)[0]].reset_index(drop=True)
    behav_info = behav_info[BEHAV_INFO_COL]

    # create ts object and concate info
    ev = nap.Ts(t[behav[event_name] == True], time_units="s")
    info = event_info.join(behav_info, how="inner")
    return ev, info


def get_index_from_info(ev_info, by=["animal", "session"], out_pos=False):
    """
    Get a dictionary of {condition: indices} from the event info dataframe with the specified grouping columns.
    Parameters:
    - ev_info: The event information (pandas DataFrame).
    - by: List of column names to group by (default is ["animal", "session", "trial"]).
    - out_pos: If True, returns the positions of the indices instead of the label indices.

    Returns:
    - A pandas Index object containing the unique combinations of specified columns.
    """
    if out_pos:
        ev_info = ev_info.reset_index(drop=True)
    grps = ev_info.groupby(by, sort=False, dropna=False)
    index_dict = {name: group.index for name, group in grps}
    return index_dict


def compute_perievent(
    spike_count,
    behav,
    info,
    win=(-0.5, 1.0),
    bin_size=0.03,
    ev_names=["EventGo_cue", "EventTarget_Onset"],
    ev_short=["go", "target"],
):
    """
    Compute perievent spike counts for each neuron around the specified event timestamps.

    Parameters:
    - spike_count: The spike count data (nap.TsdFrame).
    - behav: The behavioral data (nap.TsdFrame).
    - info: The info data (pandas DataFrame).
    - win: A tuple specifying the time window around the event (start, end).
    - bin_size: The size of the time bins for counting spikes (default is 0.03 seconds).
    - ev_names: A list of event names to compute perievent spike counts for (default is ["EventGo_cue", "EventTarget_Onset"]).
    - ev_short: A list of short names for the events (default is ["go", "target"]).

    Returns:
    - A nap.TsdFrame containing the perievent spike counts for each neuron.
    """
    out = {}

    # compute a coefficient to rescale bin_average to bin_count
    bs_prev = np.mean(np.diff(spike_count.times()))
    coef = bin_size / bs_prev

    for name, short in zip(ev_names, ev_short):
        ev, ev_info = get_event_timestamps(behav, info, event_name=name)
        out[f"info_{short}"] = ev_info
        out[f"ev_{short}"] = ev

        # compute peth, and bin average then we calculate a coefficient to scale the binned peth to get the bin_count
        peth = coef * nap.compute_perievent(spike_count, ev, window=win).bin_average(
            bin_size=bin_size, time_units="s"
        )
        out[f"peth_{short}"] = peth
        out[f"bpeth_{short}"] = nap.compute_perievent(
            behav[BEHAV_VAR_COL], ev, window=win
        ).bin_average(bin_size=bin_size, time_units="s")
        out[f"bpeth_{short}_columns"] = BEHAV_VAR_COL

    return out
