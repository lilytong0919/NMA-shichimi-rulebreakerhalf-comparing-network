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


# ····································································
# :       _  _              _      _ __                              :
# :      | || |    ___     | |    | '_ \   ___      _ _    ___       :
# :      | __ |   / -_)    | |    | .__/  / -_)    | '_|  (_-<       :
# :      |_||_|   \___|   _|_|_   |_|__   \___|   _|_|_   /__/_      :
# :     _|"""""|_|"""""|_|"""""|_|"""""|_|"""""|_|"""""|_|"""""|     :
# :     "`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'     :
# ····································································
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


# ····················································································
# :         ___                     _                                      _         :
# :        | _ \   ___      _ _    (_)     ___    __ __    ___    _ _     | |_       :
# :        |  _/  / -_)    | '_|   | |    / -_)   \ V /   / -_)  | ' \    |  _|      :
# :       _|_|_   \___|   _|_|_   _|_|_   \___|   _\_/_   \___|  |_||_|   _\__|      :
# :     _| """ |_|"""""|_|"""""|_|"""""|_|"""""|_|"""""|_|"""""|_|"""""|_|"""""|     :
# :     "`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'     :
# ····················································································
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


# ····································
# :         ___    ___     ___       :
# :        | _ \  / __|   /   \      :
# :        |  _/ | (__    | - |      :
# :       _|_|_   \___|   |_|_|      :
# :     _| """ |_|"""""|_|"""""|     :
# :     "`-0-0-'"`-0-0-'"`-0-0-'     :
# ····································
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes


def get_top_n_PC(peth, npc=10):
    """
    Fit PCA to one session's PETH data.

    Parameters
    ----------
    peth
        nap.TsdTensor with shape: (n_time, n_event, n_cell)

    npc
        Number of desired principal components.

    Returns
    -------
    pc_scores : np.ndarray
        PCA scores with shape:
        (n_time, n_event, npc)

    pca : sklearn.decomposition.PCA
        Fitted PCA model.
    """
    nt, nev, ncell = peth.shape

    if npc > ncell:
        raise ValueError(f"npc={npc} is too large. Maximum allowed is " f"{ncell}.")

    # Convert from time × event × cell
    # to observation × cell
    peth_dat = np.asarray(peth.data()).reshape(-1, ncell)

    # Replace nonfinite values
    peth_dat = np.nan_to_num(
        peth_dat,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    pca = PCA(n_components=npc)

    pc_scores = pca.fit_transform(peth_dat)
    pc_scores = pc_scores.reshape(nt, nev, npc)

    return pc_scores, pca


def align_session_pcs(session_pc_scores, reference_index=0):
    """
    Align separately fitted session PCA spaces using orthogonal Procrustes.

    Parameters
    ----------
    session_pc_scores : list of np.ndarray
        Each array has shape:
        (n_time, n_event, n_pc)

    reference_index : int
        Session used as the reference coordinate system.

    Returns
    -------
    aligned_scores : list of np.ndarray
        Aligned event-level scores.

    aligned_means : np.ndarray
        Shape:
        (n_time, n_session, n_pc)

    rotations : list of np.ndarray
        Procrustes rotation for each session.
    """
    templates = [scores.mean(axis=1) for scores in session_pc_scores]

    reference = templates[reference_index]

    # Center the template across time before fitting rotation
    reference_centered = reference - reference.mean(axis=0, keepdims=True)

    aligned_scores = []
    rotations = []

    for scores, template in zip(
        session_pc_scores,
        templates,
    ):
        template_centered = template - template.mean(axis=0, keepdims=True)

        R, _ = orthogonal_procrustes(
            template_centered,
            reference_centered,
        )

        # R operates on the final PC dimension
        scores_aligned = scores @ R

        aligned_scores.append(scores_aligned)
        rotations.append(R)

    return aligned_scores, rotations


def convert_peth_to_pcpeth(ev_peth, ev_info, npc=10):
    # Placeholder for the actual implementation
    sess_idx = get_index_from_info(ev_info, ["session"], out_pos=True)
    sess_pcscores = []
    sess_infos = []
    for sess, indices in sess_idx.items():
        sess_peth = ev_peth[:, indices, :]
        sess_ev_info = ev_info.iloc[indices]
        pc_scores, pca = get_top_n_PC(sess_peth, npc=npc)
        sess_pcscores.append(pc_scores)
        sess_infos.append(sess_ev_info)
    # Align the session PCA spaces
    aligned_sess_pc, rotations = align_session_pcs(sess_pcscores, reference_index=0)
    # combine the aligned session PCA scores into a single tensor
    pc_d = np.concatenate(aligned_sess_pc, axis=1)
    pc_t = ev_peth.times()
    ev_pcpeth = nap.TsdTensor(t=pc_t, d=pc_d, time_units="s")
    return ev_pcpeth
