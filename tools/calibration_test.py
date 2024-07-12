import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

def pose_to_matrix(pose):
    """
    将位姿转换为齐次变换矩阵
    """
    translation = pose[:3]
    rotation = R.from_euler('xyz', pose[3:]).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix

def load_data():
    """
    加载相机点云数据和机械臂末端位姿数据
    """
    # 示例数据，实际应用中需要从文件或传感器读取
    camera_points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0])
    ]
    
    gripper_poses = [
        np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.1]),  # x, y, z, roll, pitch, yaw
        np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.2]),
        np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.3]),
        np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.4])
    ]
    
    return camera_points, gripper_poses

def hand_eye_calibration(camera_points, gripper_poses):
    """
    手眼标定
    """
    camera_matrices = [pose_to_matrix(pose) for pose in camera_points]
    gripper_matrices = [pose_to_matrix(pose) for pose in gripper_poses]
    
    # 使用OpenCV中的calibrateHandEye方法
    R_gripper_to_base = np.array([matrix[:3, :3] for matrix in gripper_matrices])
    t_gripper_to_base = np.array([matrix[:3, 3] for matrix in gripper_matrices])
    R_target_to_camera = np.array([matrix[:3, :3] for matrix in camera_matrices])
    t_target_to_camera = np.array([matrix[:3, 3] for matrix in camera_matrices])
    
    R_camera_to_gripper, t_camera_to_gripper = cv2.calibrateHandEye(
        R_gripper_to_base, t_gripper_to_base,
        R_target_to_camera, t_target_to_camera
    )
    
    T_camera_to_gripper = np.eye(4)
    T_camera_to_gripper[:3, :3] = R_camera_to_gripper
    T_camera_to_gripper[:3, 3] = t_camera_to_gripper
    
    return T_camera_to_gripper

if __name__ == "__main__":
    camera_points, gripper_poses = load_data()
    T_camera_to_gripper = hand_eye_calibration(camera_points, gripper_poses)
    print("Transform matrix from camera to gripper:")
    print(T_camera_to_gripper)
