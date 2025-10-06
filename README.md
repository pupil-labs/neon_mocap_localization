# Localize Neon in Vicon Coordinate System

The `localize_neon_in_vicon.py` file is meant for the [Every move you make](https://pupil-labs.com/products/neon/shop#every-move-you-make) and the [I can track clearly now](https://pupil-labs.com/products/neon/shop#i-can-track-clearly-now) frames, as well as [any custom frames](https://docs.pupil-labs.com/neon/hardware/make-your-own-frame/) that carry Motion Capture markers.

It calculates the position of Neon's scene camera in the Vicon coordinate system.
It could be modified for other MoCap systems.

While running, it produces some plots to help see what it is doing.
Simply close each plot when you are done inspecting it for the script to proceed to the next step.

The script requires the following:

- A Vicon CSV taken during a Neon recording, where Neon is looking [at AprilTags on a flat surface](https://docs.pupil-labs.com/neon/pupil-cloud/enrichments/marker-mapper/#setup).
  - The AprilTags _must_ be accompanied by MoCap IR markers that are also positioned on the same flat surface.
- An image of the AprilTags, as taken with Neon's scene camera during said Vicon recording.
- Neon's scene camera intrinsics data, as contained in [the `scene_camera.json` file](https://docs.pupil-labs.com/neon/data-collection/data-format/#scene-camera-json).

To run it, first install the requirements:

```
pip install -r requirements.txt
```

Then:

```
python localize_neon_in_vicon.py --image apriltags.png --calib scene_camera.json --vicon Calib_pupil_labs_table.csv
```

It will save a "Neon+Vicon calibration file" in `neon_camera_pose_relative_to_markers.json`. It will also print out some diagnostic data.

After you have run the localization script, you can then make use of the `map_gaze_to_vicon` function in the `map_gaze_to_vicon.py` file, as follows:

```python
from pose_vicon import Pose
from map_gaze_to_vicon import map_gaze_to_vicon

vicon_transform = []
with open("neon_camera_pose_relative_to_markers.json", "r") as f:
    vicon_transform = json.load(f)
    neon_relative_pose = Pose(
        position=vicon_transform["position"],
        rotation=vicon_transform["rotation"],
    )

map_gaze_to_vicon(
    neon_relative_pose,
    azimuth,
    elevation,
    avg_neon_marker_positions,
)
```

Azimuth & elevation are provided for each Neon gaze datum in [`gaze.csv`](https://docs.pupil-labs.com/neon/data-collection/data-format/#gaze-csv).

The `avg_neon_marker_positions` value needs to be calculated for each frame of a Vicon recording. It would be advised to interpolate either the Vicon data or the Neon data to be at the same sampling rate and then temporally sync them. You can use principles from [our Time Sync guide](https://docs.pupil-labs.com/neon/data-collection/time-synchronization/).
