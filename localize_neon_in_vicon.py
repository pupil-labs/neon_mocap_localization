import argparse
import json

import cv2
import numpy as np

from apriltags import AprilTags
from load_vicon_data import load_vicon_data
from neon_vicon import Neon
from plots_vicon import (
    plot_apriltag_and_surface_in_neon,
    plot_neon_in_surface,
    plot_neon_in_vicon,
    plot_surface_local_coordinate_system_in_vicon,
)
from pose_vicon import Pose
from rigid import fit_plane
from surface_vicon import Surface

parser = argparse.ArgumentParser(
    description="Localize Neon in Vicon coordinate system using AprilTags."
)
parser.add_argument(
    "--image", required=True, help="Path to AprilTag image (e.g., frame_0089.png)"
)
parser.add_argument(
    "--calib",
    required=True,
    help="Path to scene camera calibration JSON (e.g., scene_camera.json)",
)
parser.add_argument(
    "--vicon",
    required=True,
    help="Path to Vicon CSV data",
)
args = parser.parse_args()

img_path = args.image
calib_path = args.calib
vicon_path = args.vicon

# load apriltag image from Neon recording

img = cv2.imread(img_path)

# load scene camera calibration

scene_calib = []
with open(calib_path, "r") as f:
    scene_calib = json.load(f)

scene_camera_matrix = np.array(scene_calib["camera_matrix"])

K = scene_camera_matrix
D = np.array(scene_calib["distortion_coefficients"]).flatten()

neon = Neon(K, D)

# load vicon data

neon_marker_positions_in_vicon, display_positions_vicon = load_vicon_data(vicon_path)

all_display_tag_markers_vicon = display_positions_vicon.reshape(-1, 3)

# construct the estimated pose of the display in vicon system
centroid, rotation = fit_plane(all_display_tag_markers_vicon.T)
display_pose_vicon = Pose(
    position=centroid.flatten(),
    rotation=rotation,
)
print(display_pose_vicon)

# estimate tag size [m] from vicon data
tag_1_marker_positions_in_vicon = display_positions_vicon[0, :, :].squeeze()
tag_size = np.sqrt(
    np.sum(
        np.power(
            tag_1_marker_positions_in_vicon[1, :]
            - tag_1_marker_positions_in_vicon[2, :],
            2,
        )
    )
)
print(tag_size)

# detect apriltags in neon image
apriltags = AprilTags(neon.camera_matrix, neon.dist_coeffs, tag_size)
apriltags.detect_tags(img)
apriltags.extract_tag_poses()

# find neon's pose in each apriltag coordinate system
for pose in apriltags.tag_poses:
    neon.add_pose_in_tag(pose.inverse())

# take detected tag poses and combine them into a surface
display_surface = Surface(tag_size)
for pose in apriltags.tag_poses:
    display_surface.add_tag_pose(pose)

# build the surface from the tags
display_surface.build_surface(neon.pose_in_tags[0].position)

# find neon's pose in local surface coordinate system
# NOTE: the local surface coordinate system follows the conventions of our SVD method in `fit_plane`
# It does not follow the Optitrack rotation matrix conventions, but that is okay,
# as the end result is in the OptiTrack coordinate system.
neon.set_pose_in_surface(display_surface.pose_in_neon.inverse())

# apply surface pose in vicon to neon pose in surface coordinates to get neon camera pose in vicon coordinates
neon.calculate_pose_in_vicon(display_pose_vicon)

# plot tags and surface in neon camera coordinates as sanity check
plot_apriltag_and_surface_in_neon(
    apriltags,
    display_surface,
)

# plot neon's pose in display surface coordinates as sanity check
plot_neon_in_surface(
    neon.pose_in_surface,
    display_surface,
)

# plot surface local coordinate system in vicon space,
# as obtained via SVD, as sanity check
plot_surface_local_coordinate_system_in_vicon(
    all_display_tag_markers_vicon,
    display_pose_vicon,
)

cam_z_axis_in_vicon = neon.pose_in_vicon.rotation @ np.array([[0], [0], [1.0]])

# plot the final positions, as sanity check
plot_neon_in_vicon(
    neon,
    display_positions_vicon,
    neon_marker_positions_in_vicon,
    cam_z_axis_in_vicon,
)

# Convert cam_z_axis_in_vicon (a 3D vector) to spherical coordinates
# x: left (+), y: forward (+), z: up (+)
cam_vec = cam_z_axis_in_vicon.flatten()
x, y, z = cam_vec
r = np.linalg.norm(cam_vec)
theta = np.arccos(z / r)  # inclination from z-axis
phi = np.arctan2(y, x)  # azimuth from x-axis (left)
theta = np.degrees(theta)  # convert to degrees
phi = np.degrees(phi)  # convert to degrees

print(
    f"Camera Z axis in spherical coordinates (r, theta, phi): ({r:.3f}, {theta:.3f}, {phi:.3f})"
)

print("\nAbsolute Neon scene camera pose in Vicon coordinates:\n")
print(neon.pose_in_vicon)

# determine position of neon camera relative to frame markers
avg_neon_marker_positions = np.mean(neon_marker_positions_in_vicon, axis=1)
neon_camera_position_relative_to_markers = (
    neon.pose_in_vicon.position - avg_neon_marker_positions
)
neon_camera_pose_relative_to_markers = Pose(
    position=neon_camera_position_relative_to_markers,
    rotation=neon.pose_in_vicon.rotation,
)

# invert ("recover") and plot neon scene camera relative to markers, as sanity check
neon_recovered = Neon(K, D)
neon_recovered.pose_in_vicon = Pose(
    position=neon_camera_position_relative_to_markers + avg_neon_marker_positions,
    rotation=neon.pose_in_vicon.rotation,
)
plot_neon_in_vicon(
    neon_recovered,
    display_positions_vicon,
    neon_marker_positions_in_vicon,
    cam_z_axis_in_vicon,
)

print("\nNeon camera pose relative to frame markers (Vicon coordinates):\n")
print(neon_camera_pose_relative_to_markers)

# Export neon_camera_pose_relative_to_markers to JSON file
output = {
    "position": neon_camera_pose_relative_to_markers.position.tolist(),
    "rotation": neon_camera_pose_relative_to_markers.rotation.tolist(),
}
with open("neon_camera_pose_relative_to_markers.json", "w") as f:
    json.dump(output, f, indent=4)
print(
    "\nExported neon_camera_pose_relative_to_markers to neon_camera_pose_relative_to_markers.json"
)
