import numpy as np

class CameraPoseCalculator:
    def __init__(self, calibration_matrix, intrinsic_matrix):
        self.calibration_matrix = calibration_matrix
        self.intrinsic_matrix = intrinsic_matrix

    def calculate_camera_position(self, uv_point, depth):
        # Transform UV point to camera space
        camera_point = np.dot(np.linalg.inv(self.intrinsic_matrix), np.dot(depth, uv_point))

        # Add homogeneous coordinate
        camera_point_homogeneous = np.array(np.insert(camera_point.flatten(), 3, 1)).reshape((4, 1))

        # Calculate camera position in world coordinates
        camera_position = np.dot(self.calibration_matrix, camera_point_homogeneous)

        return camera_position


if __name__ == "__main__":
    point1_uv = np.array([379, 293, 1]).reshape((3, 1))
    depth = 0.58

    calibration_matrix = np.array([[0.82067301, -0.30800748, 0.48127664, 0.3458119],
                                    [-0.54902958, -0.19169852, 0.81352209, -0.71810042],
                                    [-0.15831087, -0.93187074, -0.32642702, 0.2831532],
                                    [0., 0., 0., 1.]])

    K = np.array([387.31561279296875, 0.0, 317.2518310546875,
                  0.0, 387.31561279296875, 243.16317749023438,
                  0.0, 0.0, 1.0]).reshape((3, 3))

    camera_pose_calculator = CameraPoseCalculator(calibration_matrix, K)

    camera_position = camera_pose_calculator.calculate_camera_position(point1_uv, depth)
    print('Camera position:', camera_position)
