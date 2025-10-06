import numpy as np
import pandas as pd


def load_vicon_data(file_path):
    vicon_data = pd.read_csv(file_path)

    l = len(vicon_data) // 2
    LTOP = np.array(
        [
            vicon_data["LTOP_X"][l],
            vicon_data["LTOP_Y"][l],
            vicon_data["LTOP_Z"][l],
        ]
    )
    LFRONT = np.array(
        [
            vicon_data["LFRONT_X"][l],
            vicon_data["LFRONT_Y"][l],
            vicon_data["LFRONT_Z"][l],
        ]
    )
    LBACK = np.array(
        [
            vicon_data["LBACK_X"][l],
            vicon_data["LBACK_Y"][l],
            vicon_data["LBACK_Z"][l],
        ]
    )

    RTOP = np.array(
        [
            vicon_data["RTOP_X"][l],
            vicon_data["RTOP_Y"][l],
            vicon_data["RTOP_Z"][l],
        ]
    )
    RFRONT = np.array(
        [
            vicon_data["RFRONT_X"][l],
            vicon_data["RFRONT_Y"][l],
            vicon_data["RFRONT_Z"][l],
        ]
    )
    RBACK = np.array(
        [
            vicon_data["RBACK_X"][l],
            vicon_data["RBACK_Y"][l],
            vicon_data["RBACK_Z"][l],
        ]
    )

    neon_marker_positions_in_vicon = np.array(
        [LTOP, LFRONT, LBACK, RTOP, RFRONT, RBACK]
    ).T

    display_positions_vicon = np.zeros((4, 4, 3))
    for i in range(4):
        display_positions_vicon[i, 0, :] = np.array(
            [
                vicon_data[f"TL{i + 1}_X"][l],
                vicon_data[f"TL{i + 1}_Y"][l],
                vicon_data[f"TL{i + 1}_Z"][l],
            ]
        )
        display_positions_vicon[i, 1, :] = np.array(
            [
                vicon_data[f"BR{i + 1}_X"][l],
                vicon_data[f"BR{i + 1}_Y"][l],
                vicon_data[f"BR{i + 1}_Z"][l],
            ]
        )
        display_positions_vicon[i, 2, :] = np.array(
            [
                vicon_data[f"BL{i + 1}_X"][l],
                vicon_data[f"BL{i + 1}_Y"][l],
                vicon_data[f"BL{i + 1}_Z"][l],
            ]
        )
        display_positions_vicon[i, 3, :] = np.array(
            [
                vicon_data[f"TR{i + 1}_X"][l],
                vicon_data[f"TR{i + 1}_Y"][l],
                vicon_data[f"TR{i + 1}_Z"][l],
            ]
        )

    return neon_marker_positions_in_vicon, display_positions_vicon
