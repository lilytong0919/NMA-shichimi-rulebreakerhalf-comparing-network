import pynapple as nap
import numpy as np


def nap_load_data(file_path):
    """
    Load data nwb file from the specified filepath using pynapple. Then
    modify the field names such that they are consistent across datasets.
    Input:
        file_path: str
            The path to the nwb file to be loaded.
    Return:
        data: pynapple.NWBFile
            The loaded nwb file as a pynapple.NWBFile object.
    """
    key_map = {
        "EventTarget_onset": "EventTarget_Onset",
        "targets_dir": "target_dir",
        "Target_ID": "target_ID",
    }
    data = nap.load_file(file_path)

    print(f"Original keys: {list(data.keys())}")
    for k, new_k in key_map.items():
        if k in data.keys():
            print(f"Renaming key '{k}' to '{new_k}'")
            data[new_k] = data.pop(k)
    # Need to update an hidden attribute _view to reflect the changes in keys
    # when doing print(data). Else the printed view will show the old keys.
    data._view = [
        [
            key,
            (
                value["type"]
                if isinstance(value, dict) and "type" in value
                else type(value).__name__
            ),
        ]
        for key, value in data.data.items()
    ]
    return data


def get_behav(data: nap.NWBFile) -> nap.TsdFrame:
    """
    Get the behavioral data from the pynapple.NWBFile object, load in
    the actual data from the pynapple.NWBFile object, and return it as a
    pynapple.TsdFrame object. Within this function we defined a list of
    behavioral keys, if we cannot find correponding keys in the pynapple.NWBFile
    object, we will fill the data with NaN.

    Input:
        data: pynapple.NWBFile
            The loaded nwb file as a pynapple.NWBFile object.
    Return:
        behav_data: pynapple.TsdFrame
            The behavioral data from the loaded nwb file.
    """
    behav_keys = [
        "EventGo_cue",
        "EventTarget_Onset",  # trigger times
        "target_ID",
        "target_dir",
        "result",  # cue identity, and behav outcome
        "cursor_pos_x",
        "cursor_pos_y",
        "cursor_vel_x",
        "cursor_vel_y",
        "cursor_acc_x",
        "cursor_acc_y",  # cursor kinematics
    ]
    d = []
    t = data[behav_keys[0]].times()
    for k in behav_keys:
        if k in data.keys():
            d.append(data[k].data()[:])
            if not np.all(np.isclose(t, data[k].times())):
                raise ValueError(f"Times for {k} do not match the first key's times.")
        else:
            d.append(np.full_like(t, np.nan))  # Fill with NaN if key is missing
    data = nap.TsdFrame(t, np.column_stack(d), columns=behav_keys)
    return data


def get_spike_count(data: nap.NWBFile):
    """
    Get the spikes counts from the pynapple.NWBFile object, load in
    the actual data from the pynapple.NWBFile object, and return it as a pynapple.TsdFrame object.
    Input:
        data: pynapple.NWBFile
            The loaded nwb file as a pynapple.NWBFile object.
    Return:
        spike_count: pynapple.TsdFrame
            The spike count from the loaded nwb file.
    """
    spike_count = data["spikes_counts"].copy()
    return spike_count


def get_info(data: nap.NWBFile):
    """
    Get the info data from the pynapple.NWBFile object, load in
    the actual data from the pynapple.NWBFile object, and return it as a pynapple.TsdFrame object.
    Input:
        data: pynapple.NWBFile
            The loaded nwb file as a pynapple.NWBFile object.
    Return:
        info: pynapple.TsdFrame
            The info data from the loaded nwb file.
    """
    info_keys = ["trial_id", "animal", "datasetID", "session", "brain_region"]
    d = []
    t = data[info_keys[0]].times()
    for k in info_keys:
        if k in data.keys() or k == "brain_region":
            # if key is brain region we figureout the brain region from the datasetID
            if k == "brain_region":
                datasetID = data["datasetID"].data()[0]
                if (
                    datasetID == 3
                ):  # dataset 3 is the only dataset that has brain region info
                    to_append = data[k].data()[:]
                elif datasetID == 4:  # dataset 4 is all M1
                    to_append = np.repeat("M1", len(t))
                elif (
                    datasetID == 5
                ):  # 5 if most complicated, only animal 3 and 4 have known brain region,
                    # info obtained from original paper: https://www.nature.com/articles/s41593-019-0555-4#Sec11
                    # monkey J = animal4 = M1, monkey T = animal3 = PMd

                    # we initialize the data as nans
                    to_append = np.full_like(t, np.nan, dtype=object)
                    # then check for animal id, replace corresponding position with known brain region
                    bw3 = data["animal"].to_numpy() == 3
                    bw4 = data["animal"].to_numpy() == 4
                    to_append[bw3] = "PMd"
                    to_append[bw4] = "M1"

                else:
                    to_append = np.full_like(t, np.nan, dtype=object)
            else:
                to_append = data[k].data()[:]
                # check timestamps matches if we are just getting the data from the nwb file, if not we raise an error
                if not np.all(np.isclose(t, data[k].times())):
                    raise ValueError(
                        f"Times for {k} do not match the first key's times."
                    )
            # try to deal with bytestring, if the data is bytestring we decode it to string
            if isinstance(to_append[0], bytes):
                to_append = np.array([x.decode("utf-8") for x in to_append])
            # append the data to the list
            d.append(to_append)
        else:
            d.append(np.full_like(t, np.nan))  # Fill with NaN if key is missing
    data = nap.TsdFrame(t, np.column_stack(d), columns=info_keys)
    return data
