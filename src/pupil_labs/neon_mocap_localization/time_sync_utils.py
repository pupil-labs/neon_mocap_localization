import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view


def align_signals(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    y_ts: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], list[int], np.signedinteger]:
    if np.isnan(x).any():
        x[np.isnan(x)] = np.nanmean(x)

    if np.isnan(y).any():
        y[np.isnan(y)] = np.nanmean(y)

    n = len(x)

    windows = sliding_window_view(y, n)
    sq_diff = np.sum((windows - x) ** 2, axis=1)

    max_corr_idx = np.argmin(sq_diff)

    x_time_in_y = y_ts[max_corr_idx : (max_corr_idx + len(x))]
    x_idxs_in_y = list(range(max_corr_idx, (max_corr_idx + len(x))))

    return x_time_in_y, x_idxs_in_y, max_corr_idx
