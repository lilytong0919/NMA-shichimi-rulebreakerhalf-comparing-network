import pynapple as nap
import numpy as np
from shared.utils import parse_kwargs
from scipy.signal.windows import boxcar, gaussian

# try this, set backend of pynapple to jax so we can have GPU utilization?
nap.nap_config.set_backend("numba")


# processing function for neural data.
# Work on pynapple data structures.
def make_fr(spikes: nap.TsGroup, **kwargs) -> dict[str, nap.TsdFrame]:
    # TODO: add option to input tsdFrame as well
    # Bin into certaint time bins, normalize, and smooth.
    cfg = {
        "dt": 0.05,
        "boxcar_size": 0.1,
        "smooth": None,
        "gausswin_size": 1,
        "gausswin_sd": 0.02,
        "gauss_normalize": True,
        "time_units": "s",
    }
    cfg = parse_kwargs(
        cfg, kwargs, strict=False
    )  # update cfg with kwargs, check for unkown keywords
    # potentially add logger to load config to ensure correct config has been pass down.

    time_unit = cfg["time_units"]

    boxcar_kernel = None
    smooth_kernel = None  # a list of kernels to apply for smoothing maybe?
    if cfg["boxcar_size"]:
        print(f"Creating boxcar kernel, will be applied")
        boxcar_kernel = boxcar(M=int(cfg["boxcar_size"] / cfg["dt"]))
    if cfg["smooth"] == "gauss":
        print(f"Creating gaussian kernel, will be applied")
        smooth_kernel = gaussian(
            M=int(cfg["gausswin_size"] / cfg["dt"]), std=cfg["gausswin_sd"] / cfg["dt"]
        )
        if cfg["gauss_normalize"]:
            smooth_kernel = smooth_kernel / smooth_kernel.sum()

    # Get subselection of data maybe? Or I should probably do this later
    output = {}
    fr = spikes.count(
        cfg["dt"], time_units=time_unit
    )  # /cfg["dt"]#,ep=nap.IntervalSet(my_t[0]-0.05,my_t[-1]))
    steps = fr.copy()
    if boxcar_kernel is not None:
        fr = fr.convolve(boxcar_kernel)
        moving_bin = fr.copy()
    else:
        moving_bin = None
    # zfr = (fr - np.nanmean(fr,0))/np.nanstd(fr,0)
    if smooth_kernel is not None:
        fr = fr.convolve(smooth_kernel)
        smoothed = fr.copy()
    else:
        smoothed = None

    output = {"steps": steps, "bins": moving_bin, "smoothed": smoothed}
    return output


def get_peth(
    data: nap.Tsd | nap.TsdFrame | nap.TsdTensor | nap.TsGroup,
    events: nap.IntervalSet | nap.Ts,
    win: tuple[int, int],
) -> nap.TsdTensor | dict:
    """
    Compute perievent time histogram for continuous or spike data.

    Parameters
    ----------
    data : nap.Tsd | nap.TsdFrame | nap.TsdTensor | nap.TsGroup
        Input data. If Tsd/TsdFrame/TsdTensor, uses compute_perievent_continuous.
        If TsGroup, uses compute_perievent.
    events : nap.IntervalSet | nap.Ts
        Reference events. If IntervalSet, will extract interval centers.
        If Ts/Tsd, uses timestamps directly.
        Technically pynapple allow events to be Tsd/TsdFrame/TsdTensor too. Only
        the index of Tsd/TsdFrame/TsdTensor will be used as event times. For simplicity
        and less confusion, We set a stricter requirement to only allow Ts.
    win : tuple[int, int]
        Time window around events (start, end) in seconds.

    Returns
    -------
    nap.TsdTensor | dict
        For continuous data: (time, ev, data_dimensions) TsdTensor
        For TsGroup: dict of TsdFrame per unit
    """
    # Determine event times: get center if IntervalSet, otherwise use directly
    if isinstance(events, nap.IntervalSet):
        tref = events.get_intervals_center()
    elif isinstance(events, (nap.Ts, nap.Tsd)):
        tref = events
    else:
        raise TypeError(f"events must be IntervalSet, Ts, or Tsd, got {type(events)}")

    # Choose compute function based on data type
    if isinstance(data, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
        peth = nap.compute_perievent_continuous(
            timeseries=data, tref=tref, minmax=win, time_unit="s"
        )
    elif isinstance(data, (nap.TsGroup, nap.Ts)):
        peth = nap.compute_perievent(
            timestamps=data, tref=tref, minmax=win, time_unit="s"
        )
    else:
        raise TypeError(
            f"data must be Tsd, TsdFrame, TsdTensor, or TsGroup, got {type(data)}"
        )

    return peth
