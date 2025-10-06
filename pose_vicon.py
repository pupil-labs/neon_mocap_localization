import numpy as np


class Pose:
    def __init__(self, position, rotation):
        self.position = position
        self.rotation = rotation

    def __repr__(self):
        return f"Pose(\nposition={self.position},\n\nrotation={self.rotation}\n)"

    def inverse(self):
        """Return the inverse of the pose."""
        inv_rotation = self.rotation.T
        inv_position = -inv_rotation @ self.position
        return Pose(position=inv_position, rotation=inv_rotation)

    def to_pupil_labs_vicon_format(self):
        """Convert the pose to Vicon coordinate system."""
        vicon_position = np.array(
            [self.position[0], self.position[1], -self.position[2]]
        )

        R_apriltag_to_vicon = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, -1],
            ]
        )

        vicon_rotation = self.rotation.copy()
        vicon_rotation = R_apriltag_to_vicon @ vicon_rotation

        return Pose(position=vicon_position, rotation=vicon_rotation)

    def to_matrix(self):
        """Convert the pose to a transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.position
        return matrix

    def apply(self, pose):
        """Apply this pose to another pose."""
        # new_position = self.rotation @ pose.position + self.position
        new_position = self.rotation @ pose.position + self.position
        new_rotation = self.rotation @ pose.rotation

        return Pose(position=new_position, rotation=new_rotation)
