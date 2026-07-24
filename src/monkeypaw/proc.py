import numpy as np
import pandas as pd
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


def find_incorrectly_labeled_rewarded_trials(df,
                                             pos_threshold_cm=2,
                                             duration_expected_s=3.0,
                                             angle_tolerance_deg=20):
    """
    Identifies trials labeled as 'Rewarded' ('R') that violate specified criteria,
    considering each brain region segment (if present) separately within a trial.

    Criteria:
    1. 'EventTarget_Onset' or 'EventGo_cue' cursor positions are outside
       a specified radial threshold from (0,0).
    2. Duration from 'EventGo_cue' to the end of the trial segment is not within
       (duration_expected).
    3. The cursor's endpoint direction for the trial segment is not within angle_tolerance_deg of the target_dir.
    4. The 'target_dir' value is missing (NaN).

    Args:
        df (pd.DataFrame): The DataFrame containing trial data.
        pos_threshold_cm (int): Max distance from (0,0) for target onset/go cue.
        duration_expected_s (float): Expected duration from go cue to the end of the trial segment.
        angle_tolerance_deg (float): Allowed deviation for endpoint angle from target direction.

    Returns:
        pd.DataFrame: A DataFrame listing 'Rewarded' trials that violate any criteria,
                      organised by animal, session, trial_id, and detailing if any brain regions
                      caused the violation, along with event presence.
    """
    all_incorrect_segments = []
    angle_tolerance_rad = np.radians(angle_tolerance_deg)
    max_expected_duration = duration_expected_s

    # Group by trial_id and brain_region to evaluate each segment separately
    # dropna=False ensures NaN brain_regions are treated as a distinct group
    grouped_segments = df.groupby(['animal', 'session', 'trial_id', 'brain_region'], dropna=False)

    for group_keys, segment_df in grouped_segments:
        animal, session, trial_id, brain_region = group_keys

        # Only process trial segments labeled as 'Rewarded'
        if 'R' not in segment_df['result'].unique():
            continue

        is_incorrect_segment = False
        segment_reasons = []

        # Initialize metrics for this trial segment
        dist_target_onset_seg = np.nan
        dist_go_cue_seg = np.nan
        actual_duration_from_go_cue_seg = np.nan
        endpoint_angle_seg_deg = np.nan
        target_direction_seg_deg = np.nan

        # Determine event presence
        target_onset_event_rows = segment_df[segment_df['EventTarget_Onset'] == True]
        go_cue_event_rows = segment_df[segment_df['EventGo_cue'] == True]
        has_target_onset_event_seg = not target_onset_event_rows.empty
        has_go_cue_event_seg = not go_cue_event_rows.empty

        # Initialize flags for comparison logic
        target_onset_dist_valid = False
        is_target_onset_within_criteria = False
        go_cue_dist_valid = False
        is_go_cue_within_criteria = False

        # --- Target Direction missing/NaN ---
        if 'target_dir' not in segment_df.columns or pd.isna(segment_df['target_dir'].iloc[0]):
            is_incorrect_segment = True
            segment_reasons.append("Target direction (target_dir) is missing or NaN")

        # Check if cursor position columns exist
        cursor_columns_exist = 'cursor_pos_x' in segment_df.columns and 'cursor_pos_y' in segment_df.columns

        if not cursor_columns_exist:
            segment_reasons.append("Cursor position (x,y) columns missing for segment")
        else:
            # Process Target Onset event
            if has_target_onset_event_seg:
                target_onset_x = target_onset_event_rows['cursor_pos_x'].iloc[0]
                target_onset_y = target_onset_event_rows['cursor_pos_y'].iloc[0]

                if np.isnan(target_onset_x) and np.isnan(target_onset_y):
                    is_incorrect_segment = True
                    segment_reasons.append("Target Onset: Both cursor_pos_x and cursor_pos_y are NaN")
                elif np.isnan(target_onset_x):
                    is_incorrect_segment = True
                    segment_reasons.append(f"Target Onset: cursor_pos_x is NaN (cursor_pos_y={target_onset_y:.2f})")
                elif np.isnan(target_onset_y):
                    is_incorrect_segment = True
                    segment_reasons.append(f"Target Onset: cursor_pos_y is NaN (cursor_pos_x={target_onset_x:.2f})")
                else: # Both are not NaN, calculate distance and check criteria
                    dist_target_onset_seg = np.sqrt(target_onset_x**2 + target_onset_y**2)
                    target_onset_dist_valid = True # Mark as valid for comparison
                    if dist_target_onset_seg > pos_threshold_cm:
                        is_incorrect_segment = True
                        segment_reasons.append(f"Target Onset dist ({dist_target_onset_seg:.2f}cm) > {pos_threshold_cm}cm")
                        is_target_onset_within_criteria = False
                    else:
                        is_target_onset_within_criteria = True

            # Process Go Cue event
            if has_go_cue_event_seg:
                go_cue_x = go_cue_event_rows['cursor_pos_x'].iloc[0]
                go_cue_y = go_cue_event_rows['cursor_pos_y'].iloc[0]

                if np.isnan(go_cue_x) and np.isnan(go_cue_y):
                    is_incorrect_segment = True
                    segment_reasons.append("Go Cue: Both cursor_pos_x and cursor_pos_y are NaN")
                elif np.isnan(go_cue_x):
                    is_incorrect_segment = True
                    segment_reasons.append(f"Go Cue: cursor_pos_x is NaN (cursor_pos_y={go_cue_y:.2f})")
                elif np.isnan(go_cue_y):
                    is_incorrect_segment = True
                    segment_reasons.append(f"Go Cue: cursor_pos_y is NaN (cursor_pos_x={go_cue_x:.2f})")
                else: # Both are not NaN, calculate distance and check criteria
                    dist_go_cue_seg = np.sqrt(go_cue_x**2 + go_cue_y**2)
                    go_cue_dist_valid = True # Mark as valid for comparison
                    if dist_go_cue_seg > pos_threshold_cm:
                        is_incorrect_segment = True
                        segment_reasons.append(f"Go Cue dist ({dist_go_cue_seg:.2f}cm) > {pos_threshold_cm}cm")
                        is_go_cue_within_criteria = False
                    else:
                        is_go_cue_within_criteria = True

            # Check if one event's position is within criteria and the other's is outside
            if target_onset_dist_valid and go_cue_dist_valid:
                if (is_target_onset_within_criteria and not is_go_cue_within_criteria) or \
                   (not is_target_onset_within_criteria and is_go_cue_within_criteria):
                    is_incorrect_segment = True
                    reason_str = f"Target Onset (dist={dist_target_onset_seg:.2f}cm) is " + \
                                 ("within" if is_target_onset_within_criteria else "outside") + \
                                 f" criteria, while Go Cue (dist={dist_go_cue_seg:.2f}cm) is " + \
                                 ("within" if is_go_cue_within_criteria else "outside") + \
                                 f" criteria (threshold: {pos_threshold_cm}cm)."
                    segment_reasons.append(reason_str)

        # --- Criterion 2: Duration from Go Cue to End of Trial Segment --- (requires Go_cue event)
        # may add duration from target_onset event to end of trial segment later
        if has_go_cue_event_seg:
            go_cue_time = go_cue_event_rows['time_s'].iloc[0]
            # 'End of trial segment' means the last timestamp in this specific segment_df
            segment_end_time = segment_df['time_s'].iloc[-1]
            actual_duration_from_go_cue_seg = segment_end_time - go_cue_time
            if not (actual_duration_from_go_cue_seg <= max_expected_duration):
                is_incorrect_segment = True
                segment_reasons.append(f"Go Cue to End of trial duration ({actual_duration_from_go_cue_seg:.2f}s) higher than {max_expected_duration:.2f}s")
        else:
            segment_reasons.append("Go Cue event missing, cannot check duration")

        # --- Criterion 3: Endpoint direction relative to Target Direction ---
        # Ensure cursor position columns exist for this check as well
        if cursor_columns_exist and 'target_dir' in segment_df.columns and not pd.isna(segment_df['target_dir'].iloc[0]):
            endpoint_x = segment_df['cursor_pos_x'].iloc[-1]
            endpoint_y = segment_df['cursor_pos_y'].iloc[-1]
            target_direction_rad = segment_df['target_dir'].iloc[0]

            if np.isnan(endpoint_x) or np.isnan(endpoint_y):
                is_incorrect_segment = True
                segment_reasons.append("Endpoint cursor position (x or y) is NaN for angle check")
            else:
                endpoint_angle_rad = np.arctan2(endpoint_y, endpoint_x)

                # Calculate angular difference, normalizing to -pi to pi
                angle_diff_rad = np.arctan2(np.sin(endpoint_angle_rad - target_direction_rad), np.cos(endpoint_angle_rad - target_direction_rad))

                endpoint_angle_seg_deg = np.degrees(endpoint_angle_rad)
                target_direction_seg_deg = np.degrees(target_direction_rad)

                if np.abs(angle_diff_rad) > angle_tolerance_rad:
                    is_incorrect_segment = True
                    segment_reasons.append(f"Endpoint angle ({endpoint_angle_seg_deg:.1f}°) not ~Target ({target_direction_seg_deg:.1f}°, diff {np.degrees(angle_diff_rad):.1f}°)")
        else:
            # This else block is for when cursor_columns_exist is False or target_dir is missing/NaN
            # The specific NaN check for target_dir is now handled at the beginning of the loop.
            if not cursor_columns_exist:
                 segment_reasons.append("Cursor data missing for angle check (for endpoint direction)")
            # If target_dir is NaN, it's already added a reason at the start.

        # If this segment was flagged, record its details
        if is_incorrect_segment:
            all_incorrect_segments.append({
                'animal': animal,
                'session': session,
                'trial_id': trial_id,
                'brain_region_flagged': brain_region if pd.notna(brain_region) else 'NaN',
                'result_label': 'R',
                'target_onset_dist_cm': dist_target_onset_seg,
                'go_cue_dist_cm': dist_go_cue_seg,
                'duration_from_go_cue_s': actual_duration_from_go_cue_seg,
                'endpoint_angle_deg': endpoint_angle_seg_deg,
                'target_direction_deg': target_direction_seg_deg,
                'has_target_onset_event': has_target_onset_event_seg,
                'has_go_cue_event': has_go_cue_event_seg,
                'reasons_seg': '; '.join(segment_reasons)
            })

    if not all_incorrect_segments:
        return pd.DataFrame() # Return empty DataFrame if no segments were incorrect

    # Convert to DataFrame of incorrect segments
    incorrect_segments_df = pd.DataFrame(all_incorrect_segments)

    # Consolidate results by trial (animal, session, trial_id)
    # If any segment within a trial is incorrect, the trial is incorrect.
    # Aggregate brain regions and reasons, taking first non-NaN metric value.
    consolidated_incorrect_trials = incorrect_segments_df.groupby(['animal', 'session', 'trial_id']).agg(
        result_label=('result_label', 'first'),
        brain_regions_in_trial=('brain_region_flagged', lambda x: ', '.join(sorted(x.unique()))),
        target_onset_dist_cm=('target_onset_dist_cm', lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan),
        go_cue_dist_cm=('go_cue_dist_cm', lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan),
        duration_from_go_cue_s=('duration_from_go_cue_s', lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan),
        endpoint_angle_deg=('endpoint_angle_deg', lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan),
        target_direction_deg=('target_direction_deg', lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan),
        has_target_onset_event=('has_target_onset_event', 'any'), # True if any segment had the event
        has_go_cue_event=('has_go_cue_event', 'any'), # True if any segment had the event
        reasons=('reasons_seg', lambda x: '; '.join(sorted(x.unique()))),
    ).reset_index()

    return consolidated_incorrect_trials
