import numpy as np
import pandas as pd
import pynapple as nap
from monkeypaw.proc import get_index_from_info, convert_peth_to_pcpeth
import rsatoolbox


def remove_info_nans(peth, info, fields=["target_deg", "result"]):
    """
    Remove rows in info and corresponding peth that have NaN values in specified fields.

    Parameters:
    peth (nap.TsdTensor): The perievent time histogram data.
    info (pd.DataFrame): The event information corresponding to the peth.
    fields (list): List of field names in info to check for NaN values.

    Returns:
    filtered_peth (nap.TsdTensor): The filtered peth with NaN rows removed.
    filtered_info (pd.DataFrame): The filtered info with NaN rows removed.
    """
    # Create a boolean mask for rows without NaN values in the specified fields
    mask = ~info[fields].isnull().any(axis=1)
    keep_indices = np.where(mask)[0]

    # Filter the info DataFrame
    filtered_info = info[mask].reset_index(drop=True)

    # Filter the peth tensor using the same mask
    filtered_peth = peth[:, keep_indices, :]

    return filtered_peth, filtered_info


def target_dir_2_target_deg(info):
    """
    Round the values in specified fields of the info DataFrame to the nearest integer.

    Parameters:
    info (pd.DataFrame): The event information DataFrame.
    fields (list): List of field names in info to round.

    Returns:
    rounded_info (pd.DataFrame): The info DataFrame with specified fields rounded.
    """
    target_dir = info["target_dir"].astype(float).values
    target_deg = np.round(np.degrees(target_dir), 0)
    info_out = info.copy()
    # make target degree start at 0
    target_deg = target_deg + abs(np.nanmin(target_deg))
    info_out["target_deg"] = target_deg
    return info_out


def remove_invalid_degree(
    peth, info, valid_degrees=[0, 45, 90, 135, 180, 225, 270, 315]
):
    """
    Remove rows in info that have target_deg values not in the valid_degrees list.

    Parameters:
    peth (nap.TsdTensor): The perievent time histogram data.
    info (pd.DataFrame): The event information DataFrame.
    valid_degrees (list): List of valid target_deg values.

    Returns:
    filtered_peth (nap.TsdTensor): The peth tensor with corresponding rows removed.
    filtered_info (pd.DataFrame): The info DataFrame with invalid target_deg rows removed.
    """
    mask = info["target_deg"].isin(valid_degrees)
    filtered_info = info[mask].reset_index(drop=True)
    filtered_peth = peth[:, mask.values, :]
    return filtered_peth, filtered_info


def make_rdm_datasets(peth: nap.TsdTensor, ev_info: pd.DataFrame, **kwargs):
    """
    Make RDM datasets from perievent data and event info.

    Parameters
    ----------
    peth : nap.TsdTensor
        Perievent time series data.
    ev_info : pd.DataFrame
        Event information corresponding to the perievent data.
    **kwargs : dict
        Additional keyword arguments for filtering or processing the data.

    Returns
    -------
    rdm_datasets : dict
        A dictionary containing RDM datasets for different conditions or groups.
    """

    # Break down the peth and ev_info into segment defined by animal
    animal_idx = get_index_from_info(
        ev_info, ["datasetID", "animal", "brain_region"], out_pos=True
    )
    print(
        f"Found {len(animal_idx)} unique animal-dataset-brain_region combinations, processing each separately."
    )
    for key, indices in animal_idx.items():
        datasetID, animal, brain_region = key
        print(
            f"Animal: {animal}, Brain Region: {brain_region}, Dataset ID: {datasetID}, Number of Trials: {len(indices)}"
        )

    # Example implementation (to be customized based on specific requirements)
    rdm_datasets = {k: None for k in animal_idx.keys()}

    # loop through animal and make each animal a dataset
    for key, indices in animal_idx.items():
        animal_peth = peth[:, indices, :]
        animal_ev_info = ev_info.iloc[indices]
        datasetID, animal, brain_region = key
        print(
            f"Processing animal: {animal}, brain region: {brain_region} with {len(indices)} trials."
        )
        print(f"peth shape: {animal_peth.shape}, ev_info shape: {animal_ev_info.shape}")

        # Check what brain region we are at, if more than 1 in 1 animal, we drop the animal
        # We also drop Area2
        brain_regions = np.unique(animal_ev_info["brain_region"])
        dataset_id = animal_ev_info["datasetID"].iloc[0]
        if len(brain_regions) > 1 or "Area2" in brain_regions:
            print(
                f"Animal {animal} has multiple brain regions or contains Area2. Skipping this animal."
            )
            continue
        elif "PMd" in brain_regions:
            npc = 15
            print(
                f"Setting npc to 15 for animal {animal}, brain region {brain_region}."
            )
        elif "M1" in brain_regions:
            npc = 10
            print(
                f"Setting npc to 10 for animal {animal}, brain region {brain_region}."
            )
        else:
            print(
                f"Animal {animal} has an unrecognized brain region: {brain_regions}. Skipping this animal."
            )
            continue

        # TODO: preprocessing (i.e. normalization, smoothing, etc.)
        animal_peth = animal_peth  # Placeholder for actual preprocessing logic

        # session level processsing (get PCA, and rebase to common base within animal)
        # Placeholder for actual session-level processing logic
        pcpeth = convert_peth_to_pcpeth(animal_peth, animal_ev_info, npc=npc)
        print(
            f"Converted peth with shape {animal_peth.shape} to pcpeth with shape {pcpeth.shape}."
        )

        # Convert target_dir to target_deg
        animal_ev_info = target_dir_2_target_deg(
            animal_ev_info
        )  # print statement is inside the function
        unique_degrees = np.unique(animal_ev_info["target_deg"])
        print(f"Converted target_dir to target_deg in info DataFrame, ")
        print(f"    unique target_deg values after conversion: {unique_degrees}")

        # TODO: Filter events based on specific criteria (if needed)
        filtered_peth, filtered_info = remove_info_nans(pcpeth, animal_ev_info)
        print(
            f"Filtered out NaN values, resulting in peth with shape {filtered_peth.shape}, info with shape {filtered_info.shape}."
        )
        filtered_peth, filtered_info = remove_invalid_degree(
            filtered_peth, filtered_info
        )

        # quick index to remove any unreward events
        result = filtered_info["result"].values
        keep = np.where(result == "R")[0]
        filtered_peth, filtered_info = (
            filtered_peth[:, keep, :],
            filtered_info.iloc[keep],
        )
        print(
            f"Filtered out {len(result) - len(keep)} unrewarded events, resulting in peth with shape {filtered_peth.shape}, info with shape {filtered_info.shape}."
        )

        # TODO: take average by target direction
        rep_bin = peth_to_repbin(filtered_peth, rep_win=(0, 0.5))
        obs_descrip = {
            "target_dir": filtered_info["target_deg"].values,
            "target_ID": filtered_info["target_ID"].values,
            "result": filtered_info["result"].values,
        }
        rdm_dataset = rsatoolbox.data.Dataset(
            rep_bin,
            obs_descriptors=obs_descrip,
            descriptors={"animal": animal, "brain_region": brain_region},
        )

        rdm_datasets[key] = rdm_dataset
    # clear out empty datasets
    rdm_datasets = {k: v for k, v in rdm_datasets.items() if v is not None}
    return rdm_datasets


def peth_to_repbin(peth, rep_win=(0, 0.5)):
    """
    Convert perievent time series data to representative bin data by averaging over a specified time window.

    Parameters
    ----------
    peth : nap.TsdTensor
        Perievent time series data.
    rep_win : tuple
        Time window (start, end) in seconds for averaging the peth data.

    Returns
    -------
    rep_bin : np.ndarray
        Representative bin data obtained by averaging the peth over the specified time window.
    obs_descrip : dict
        Dictionary containing observation descriptors corresponding to the representative bin data.
    """
    rep_bin = peth.restrict(nap.IntervalSet(start=rep_win[0], end=rep_win[1])).mean(
        axis=0
    )

    return rep_bin
