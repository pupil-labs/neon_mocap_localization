import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.interpolate import interp1d

from pupil_labs.neon_mocap_localization import time_sync_utils
from pupil_labs.neon_mocap_localization.qualisys import data_loader


class MainWindow(QMainWindow):
    def __init__(self, qualisys_rec, mat_export_dir, xdf_export_dir):
        super().__init__()

        self.qualisys_rec = qualisys_rec
        self.mat_export_dir = mat_export_dir
        self.xdf_export_dir = xdf_export_dir

        self.setWindowTitle("Qualisys <-> Neon Synchronization Tool")

        # Window dimensions
        geometry = self.screen().availableGeometry()
        self.setFixedSize(geometry.width() * 0.8, geometry.height() * 0.49)

        self.main_layout = QHBoxLayout()
        self.central_widget = QWidget()

        self._prepare_qualisys_window()
        self._prepare_neon_window()
        self._prepare_export_button()

        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

    def _prepare_qualisys_window(self):
        self.qtm_fig = Figure(figsize=(5, 4), dpi=100)
        self.qtm_canvas = FigureCanvas(self.qtm_fig)
        self.qtm_axes = self.qtm_fig.add_subplot(111)

        self.qtm_toolbar = NavigationToolbar2QT(self.qtm_canvas, self)

        self.qtm_begin_slider = QSlider(Qt.Horizontal)
        self.qtm_begin_slider.setMinimum(0)
        self.qtm_begin_slider.setMaximum(5000)
        self.qtm_begin_slider.setValue(0)
        self.qtm_begin_slider.valueChanged.connect(self.update_qualisys_trim_begin)
        self.qtm_begin_label = QLabel("Trim begin: 0")

        self.qtm_end_slider = QSlider(Qt.Horizontal)
        self.qtm_end_slider.setMinimum(0)
        self.qtm_end_slider.setMaximum(5000)
        self.qtm_end_slider.setValue(0)
        self.qtm_end_slider.valueChanged.connect(self.update_qualisys_trim_end)
        self.qtm_end_label = QLabel("Trim end: 0")

        self.qtm_layout = QVBoxLayout()
        self.qtm_layout.addWidget(self.qtm_toolbar)
        self.qtm_layout.addWidget(self.qtm_canvas)

        self.qtm_layout.addWidget(self.qtm_begin_label)
        self.qtm_layout.addWidget(self.qtm_begin_slider)

        self.qtm_layout.addWidget(self.qtm_end_label)
        self.qtm_layout.addWidget(self.qtm_end_slider)

        self.main_layout.addLayout(self.qtm_layout)

        self.update_qualisys_plot()

    @Slot(int)
    def update_qualisys_trim_begin(self, value):
        self.qtm_begin_label.setText(f"Trim begin: {value}")
        self.qualisys_rec.qtm_trim_begin = value

        self.update_qualisys_plot()

    @Slot(int)
    def update_qualisys_trim_end(self, value):
        self.qtm_end_label.setText(f"Trim end: {value}")
        self.qualisys_rec.qtm_trim_end = -(value + 1)

        self.update_qualisys_plot()

    def update_qualisys_plot(self):
        self.qtm_axes.clear()

        self.qtm_axes.set_title("Qualisys Data")
        self.qtm_axes.set_xlabel("LSL Time [s]")
        self.qtm_axes.set_ylabel("X position [mm]")

        self.qualisys_rec.qtm_data_for_alignment = self.qualisys_rec.qtm_data_resampled[
            self.qualisys_rec.qtm_trim_begin : self.qualisys_rec.qtm_trim_end
        ]

        _, _, offs = time_sync_utils.align_signals(
            self.qualisys_rec.qtm_data_for_alignment,
            self.qualisys_rec.xdf_data_for_alignment,
            self.qualisys_rec.xdf_ts_for_alignment,
        )

        self.qualisys_rec.aligned_qtm_ts = (
            self.qualisys_rec.qtm_reference_timestamps
            + self.qualisys_rec.xdf_ts_for_alignment[
                offs - self.qualisys_rec.qtm_trim_begin
            ]
        )

        self.qtm_axes.plot(
            self.qualisys_rec.xdf_reference_timestamps,
            self.qualisys_rec.xdf_reference_positions[:, 0].squeeze(),
        )
        self.qtm_axes.plot(
            self.qualisys_rec.aligned_qtm_ts,
            self.qualisys_rec.qtm_reference_positions[0, :].squeeze(),
        )

        mid_idx = len(self.qualisys_rec.aligned_qtm_ts) // 2
        mid_ts = self.qualisys_rec.aligned_qtm_ts[mid_idx]
        self.qtm_axes.set_xlim(mid_ts - 0.5, mid_ts + 0.5)
        self.qtm_axes.set_ylim(
            np.min(self.qualisys_rec.qtm_data_resampled),
            np.max(self.qualisys_rec.qtm_data_resampled),
        )

        self.qtm_canvas.draw()

    def _prepare_neon_window(self):
        self.neon_fig = Figure(figsize=(5, 4), dpi=100)
        self.neon_canvas = FigureCanvas(self.neon_fig)
        self.neon_axes = self.neon_fig.add_subplot(111)

        self.neon_toolbar = NavigationToolbar2QT(self.neon_canvas, self)

        self.neon_begin_slider = QSlider(Qt.Horizontal)
        self.neon_begin_slider.setMinimum(0)
        self.neon_begin_slider.setMaximum(5000)
        self.neon_begin_slider.setValue(0)
        self.neon_begin_slider.valueChanged.connect(self.update_neon_trim_begin)
        self.neon_begin_label = QLabel("Trim begin: 0")

        self.neon_end_slider = QSlider(Qt.Horizontal)
        self.neon_end_slider.setMinimum(0)
        self.neon_end_slider.setMaximum(5000)
        self.neon_end_slider.setValue(0)
        self.neon_end_slider.valueChanged.connect(self.update_neon_trim_end)
        self.neon_end_label = QLabel("Trim end: 0")

        self.neon_layout = QVBoxLayout()
        self.neon_layout.addWidget(self.neon_toolbar)
        self.neon_layout.addWidget(self.neon_canvas)

        self.neon_layout.addWidget(self.neon_begin_label)
        self.neon_layout.addWidget(self.neon_begin_slider)

        self.neon_layout.addWidget(self.neon_end_label)
        self.neon_layout.addWidget(self.neon_end_slider)

        self.main_layout.addLayout(self.neon_layout)

        self.update_neon_plot()

    @Slot(int)
    def update_neon_trim_begin(self, value):
        self.neon_begin_label.setText(f"Trim begin: {value}")
        self.qualisys_rec.neon_trim_begin = value

        self.update_neon_plot()

    @Slot(int)
    def update_neon_trim_end(self, value):
        self.neon_end_label.setText(f"Trim end: {value}")
        self.qualisys_rec.neon_trim_end = -(value + 1)

        self.update_neon_plot()

    def update_neon_plot(self):
        self.neon_axes.clear()

        self.neon_axes.set_title("Neon Gaze Data")
        self.neon_axes.set_xlabel("LSL Time [s]")
        self.neon_axes.set_ylabel("X position [px]")

        self.qualisys_rec.neon_gaze_for_alignment = (
            self.qualisys_rec.neon_gaze_resampled[
                self.qualisys_rec.neon_trim_begin : self.qualisys_rec.neon_trim_end
            ]
        )

        _, _, neon_offset = time_sync_utils.align_signals(
            self.qualisys_rec.neon_gaze_for_alignment,
            self.qualisys_rec.xdf_neon_data_for_alignment,
            self.qualisys_rec.xdf_gaze_ts_for_alignment,
        )

        self.qualisys_rec.aligned_gaze_ts = (
            self.qualisys_rec.neon_reference_timestamps
            - self.qualisys_rec.neon_reference_timestamps[0]
        ) + self.qualisys_rec.xdf_gaze_ts_for_alignment[
            neon_offset - self.qualisys_rec.neon_trim_begin
        ]

        self.neon_axes.plot(
            self.qualisys_rec.xdf_gaze_ts_for_alignment,
            self.qualisys_rec.xdf_neon_data_for_alignment,
        )

        self.neon_axes.plot(
            self.qualisys_rec.aligned_gaze_ts,
            self.qualisys_rec.neon_reference_gaze,
        )

        mid_idx = len(self.qualisys_rec.aligned_gaze_ts) // 2
        mid_ts = self.qualisys_rec.aligned_gaze_ts[mid_idx]
        self.neon_axes.set_xlim(mid_ts - 10, mid_ts + 10)
        self.neon_axes.set_ylim(
            np.min(self.qualisys_rec.neon_reference_gaze),
            np.max(self.qualisys_rec.neon_reference_gaze),
        )

        self.neon_canvas.draw()

    def _prepare_export_button(self):
        self.export_button = QPushButton("Export Synced Data")
        self.exported_label = QLabel("")

        self.export_button.clicked.connect(self.export_data)

        self.button_layout = QVBoxLayout()
        self.button_layout.addWidget(self.export_button)
        self.button_layout.addWidget(self.exported_label)

        self.main_layout.addLayout(self.button_layout)

    @Slot()
    def export_data(self):
        self.export_mocap_neon_synced_csv()
        self.export_gaze_time_in_mocap_time()
        self.export_imu_in_xdf_time()

        self.exported_label.setText("Data export finished!")

    def export_mocap_neon_synced_csv(self):
        marker_df = pd.DataFrame()
        marker_df["timestamp [ns]"] = self.qualisys_rec.neon_rec.gaze.time
        for marker_name in self.qualisys_rec.marker_names:
            marker = str(marker_name[0])
            index = self.qualisys_rec.marker_indices[marker]

            marker_pos = self.qualisys_rec.qtm_marker_positions[index].squeeze()

            if marker_pos.shape[1] == 4:
                marker_pos = marker_pos.T

            # re-interpolate qtm data to correspond exactly to neon data

            fx = interp1d(
                self.qualisys_rec.aligned_qtm_ts,
                marker_pos[0, :],
                bounds_error=False,
                fill_value=np.nan,
            )
            fy = interp1d(
                self.qualisys_rec.aligned_qtm_ts,
                marker_pos[1, :],
                bounds_error=False,
                fill_value=np.nan,
            )
            fz = interp1d(
                self.qualisys_rec.aligned_qtm_ts,
                marker_pos[2, :],
                bounds_error=False,
                fill_value=np.nan,
            )

            marker_pos_x_in_neon = fx(self.qualisys_rec.aligned_gaze_ts)
            marker_pos_y_in_neon = fy(self.qualisys_rec.aligned_gaze_ts)
            marker_pos_z_in_neon = fz(self.qualisys_rec.aligned_gaze_ts)

        marker_df[f"{marker}_X"] = list(marker_pos_x_in_neon)
        marker_df[f"{marker}_Y"] = list(marker_pos_y_in_neon)
        marker_df[f"{marker}_Z"] = list(marker_pos_z_in_neon)

        marker_df.to_csv(
            self.mat_export_dir
            + "/"
            + Path(self.qualisys_rec.mat_path).stem
            + "_marker_positions_neon_ts.csv"
        )

    def export_gaze_time_in_mocap_time(self):
        # map neon gaze timestamps to mocap time axis for later processing

        # find the subset of gaze timestamps in LSL timeline that overlap with the QTM
        # data

        first_overlapping_gaze_idx = 0
        if self.qualisys_rec.aligned_qtm_ts[0] >= self.qualisys_rec.aligned_gaze_ts[0]:
            for idx, elem in enumerate(self.qualisys_rec.aligned_gaze_ts):
                if self.qualisys_rec.aligned_qtm_ts[0] <= elem:
                    first_overlapping_gaze_idx = idx
                    break

        last_overlapping_gaze_idx = len(self.qualisys_rec.aligned_gaze_ts)
        if (
            self.qualisys_rec.aligned_qtm_ts[-1]
            <= self.qualisys_rec.aligned_gaze_ts[-1]
        ):
            for idx, elem in enumerate(reversed(self.qualisys_rec.aligned_gaze_ts)):
                if self.qualisys_rec.aligned_qtm_ts[-1] >= elem:
                    last_overlapping_gaze_idx = (
                        len(self.qualisys_rec.aligned_gaze_ts) - idx - 1
                    )
                    break

        np.savetxt(
            self.xdf_export_dir
            + "/"
            + Path(self.qualisys_rec.mat_path).stem
            + "_first_overlapping_gaze_index.txt",
            np.array([first_overlapping_gaze_idx]),
        )

        np.savetxt(
            self.xdf_export_dir
            + "/"
            + Path(self.qualisys_rec.mat_path).stem
            + "_last_overlapping_gaze_index.txt",
            np.array([last_overlapping_gaze_idx]),
        )

        np.savetxt(
            self.xdf_export_dir
            + "/"
            + Path(self.qualisys_rec.mat_path).stem
            + "_first_overlapping_gaze_xdf_timestamp.txt",
            np.array([self.qualisys_rec.aligned_gaze_ts[first_overlapping_gaze_idx]]),
        )

        np.savetxt(
            self.xdf_export_dir
            + "/"
            + Path(self.qualisys_rec.mat_path).stem
            + "_last_overlapping_gaze_xdf_timestamp.txt",
            np.array([self.qualisys_rec.aligned_gaze_ts[last_overlapping_gaze_idx]]),
        )

        gaze_time_in_qtm_xdf = self.qualisys_rec.aligned_gaze_ts[
            first_overlapping_gaze_idx:last_overlapping_gaze_idx
        ]
        np.savetxt(
            self.xdf_export_dir
            + "/"
            + Path(self.qualisys_rec.mat_path).stem
            + "_overlapping_gaze_time_in_xdf.txt",
            gaze_time_in_qtm_xdf,
        )

        gaze_time_in_qtm = gaze_time_in_qtm_xdf - self.qualisys_rec.aligned_qtm_ts[0]
        np.savetxt(
            self.xdf_export_dir
            + "/"
            + Path(self.qualisys_rec.mat_path).stem
            + "_overlapping_gaze_time_in_qualisys.txt",
            gaze_time_in_qtm,
        )

        np.savetxt(
            self.xdf_export_dir
            + "/"
            + Path(self.qualisys_rec.mat_path).stem
            + "_first_qtm_xdf_timestamp.txt",
            np.array([self.qualisys_rec.aligned_qtm_ts[0]]),
        )

        np.savetxt(
            self.xdf_export_dir
            + "/"
            + Path(self.qualisys_rec.mat_path).stem
            + "_gaze_time_in_xdf.txt",
            self.qualisys_rec.aligned_gaze_ts,
        )

    def export_imu_in_xdf_time(self):
        f_imu_time = interp1d(
            self.qualisys_rec.neon_rec.gaze.time,
            self.qualisys_rec.aligned_gaze_ts,
            bounds_error=False,
            fill_value=np.nan,
        )

        imu_time_in_xdf = f_imu_time(self.qualisys_rec.neon_rec.imu.time)
        np.savetxt(
            self.xdf_export_dir
            + "/"
            + Path(args["mocap_mat_path"]).stem
            + "_imu_time_in_xdf.txt",
            imu_time_in_xdf,
        )


if __name__ == "__main__":
    # parse args

    parser = argparse.ArgumentParser(
        description="Converts Qualisys MoCap data to a time-synced CSV file."
    )

    parser.add_argument(
        "-m",
        "--mocap_mat_path",
        help="The path to the Qualisys data (MAT file)",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--neon_rec_path",
        help="The path to the associated Neon Native Recording Data",
        required=True,
    )
    parser.add_argument(
        "-x",
        "--xdf_path",
        required=True,
        help="The XDF file produced by LabRecorder of Lab Streaming Layer",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        help="A config file containing the parameters that remain constant between \
    sessions.",
        required=True,
    )

    args = vars(parser.parse_args())

    mocap_mat_path = args["mocap_mat_path"]
    xdf_path = args["xdf_path"]
    neon_rec_path = args["neon_rec_path"]
    config_path = args["config_path"]

    qualisys_rec = data_loader.QualisysRecording(
        xdf_path,
        None,
        mocap_mat_path,
        neon_rec_path,
        config_path,
    )

    mat_export_dir = Path(mocap_mat_path).parent
    xdf_export_dir = Path(xdf_path).parent

    app = QApplication(sys.argv)
    window = MainWindow(qualisys_rec, mat_export_dir, xdf_export_dir)
    window.show()
    sys.exit(app.exec())
