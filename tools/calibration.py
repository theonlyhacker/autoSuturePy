import calbration_tools
import numpy as np
import os
import sys
import cv2 as cv
from getKinect import KinectCapture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_robot_data(filePath):
    # f = open(filePath, 'r')
    # data = f.readlines()
    # f.close()
    # for i in range(data[0]):
    #     data = data.strip('[')
    # print(data.shape)
    # 打开文本文件
    with open(filePath, 'r') as file:
        # 逐行读取文件内容
        lines = file.readlines()
    # 创建一个空的二维数组
    data = []
    # 遍历每行数据
    for line in lines:
        row = line.strip().split()
        data.append(line.strip().split())
        
    # 打印二维数组
    print(data)


def desk_reorder(x):
    """
    将Kinect的采集的角点按照“从前往后、从左往右”的顺序排列
    :param x: 要排序的数据集
    :return: 数据集首先按照z轴降序排列, 然后对每3个点x轴进行降序
    """

    def my_sort(x): return sorted(x, key=lambda i: i[0], reverse=True)

    if x.shape[-1] == 2:
        tmp = sorted(x, key=lambda i: i[1])
        tmp = np.asarray(tmp)

        res = np.concatenate([my_sort(tmp[0:2]), my_sort(tmp[5:7]),
                              my_sort(tmp[10:12])], axis=0)
    else:
        tmp = sorted(x, key=lambda i: i[2], reverse=True)
        res = np.concatenate([my_sort(tmp[0:2]), my_sort(tmp[5:7]),
                              my_sort(tmp[10:12])], axis=0)
    return res


def box_reorder(x):
    """
    将Kinect的采集的角点按照“从前往后、从左往右”的顺序排列
    :param x: 要排序的数据集
    :return: 数据集首先按照z轴降序排列，然后对每3个点x轴进行降序
    """

    def my_sort(x): return sorted(x, key=lambda i: i[0], reverse=True)

    if x.shape[-1] == 2:
        tmp = sorted(x, key=lambda i: i[1])
        tmp = np.asarray(tmp)

        res = np.concatenate([my_sort(tmp[0:5]), my_sort(tmp[5:10]),
                              my_sort(tmp[10:15]), my_sort(tmp[15:20]),
                              my_sort(tmp[20:25]),
                              ], axis=0)
    else:
        tmp = sorted(x, key=lambda i: i[2], reverse=True)
        res = np.concatenate([my_sort(tmp[0:5]), my_sort(tmp[5:10]),
                              my_sort(tmp[10:15]), my_sort(tmp[15:20]),
                              my_sort(tmp[20:25]),
                              ], axis=0)
    return res


def kinect2robot_box(robot_pts, rotate_mat, trans, savePath):
    transformed_pts = np.matmul((robot_pts-trans.reshape([1, 3])), np.linalg.inv(rotate_mat.T))
    np.savetxt(savePath + "\\box_transformed_pts.txt", transformed_pts)
    return transformed_pts


def kinect2robot_desk(robot_pts, rotate_mat, trans, savePath):
    transformed_pts = np.matmul((robot_pts-trans.reshape([1, 3])), np.linalg.inv(rotate_mat.T))
    np.savetxt(savePath + "/desk_transformed_pts.txt", transformed_pts)
    return transformed_pts




def plot_3d_from_file(filename):
    """
    从指定的txt文件中读取数据并绘制3D图。
    
    参数:
    filename (str): 包含数据的txt文件路径。每行应包含三个数值，分别代表x, y, z坐标。
    """
    # 定义读取数据的函数
    def read_data(filename):
        x, y, z = [], [], []
        with open(filename, 'r') as file:
            for line in file:
                values = line.split()
                if len(values) == 3:
                    x.append(float(values[0]))
                    y.append(float(values[1]))
                    z.append(float(values[2]))
        return x, y, z

    # 读取数据
    x, y, z = read_data(filename)

    # 绘制3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def read_points(filename):
    """
    从文件中读取3D点数据。
    
    参数:
    filename (str): 包含3D点数据的文件路径。
    
    返回:
    numpy.ndarray: 形状为 (N, 3) 的3D点数组。
    """
    points = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) == 3:
                points.append([float(v) for v in values])
    return np.array(points)


def hand_eye_calibration(camera_points, robot_points):
    """
    计算从相机坐标系到机械臂基座坐标系的转换矩阵。
    
    参数:
    camera_points (numpy.ndarray): 相机坐标系中的3D点坐标，形状为 (N, 3)。
    robot_points (numpy.ndarray): 机械臂基座坐标系中的3D点坐标，形状为 (N, 3)。
    
    返回:
    numpy.ndarray: 4x4的转换矩阵，从相机坐标系到机械臂基座坐标系。
    """
    # 计算质心
    camera_center = np.mean(camera_points, axis=0)
    robot_center = np.mean(robot_points, axis=0)

    # 去质心
    camera_points_centered = camera_points - camera_center
    robot_points_centered = robot_points - robot_center

    # 计算H矩阵
    H = np.dot(camera_points_centered.T, robot_points_centered)

    # SVD分解
    U, S, Vt = np.linalg.svd(H)

    # 计算旋转矩阵
    R = np.dot(Vt.T, U.T)

    # 确保R是一个有效的旋转矩阵
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 计算平移向量
    t = robot_center - np.dot(R, camera_center)

    # 构建4x4转换矩阵
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t

    return transformation_matrix

def transform_point(point, transformation_matrix):
    """
    使用转换矩阵将点从相机坐标系转换到机械臂基座坐标系。
    
    参数:
    point (numpy.ndarray): 相机坐标系中的3D点，形状为 (3,)。
    transformation_matrix (numpy.ndarray): 4x4转换矩阵。
    
    返回:
    numpy.ndarray: 机械臂基座坐标系中的3D点，形状为 (3,)。
    """
    point_homogeneous = np.append(point, 1)  # 转换为齐次坐标
    transformed_point_homogeneous = np.dot(transformation_matrix, point_homogeneous)
    transformed_point = transformed_point_homogeneous[:3] / transformed_point_homogeneous[3]
    return transformed_point

def lc_eye_cal():
    filePath = "data\\points\\6-28\\judge\\"
    camera_points = read_points(filePath+"kinect_chess_3d.txt")
    robot_points = read_points(filePath+"robot_Pts.txt")
    # 计算转换矩阵
    transformation_matrix = hand_eye_calibration(camera_points, robot_points)
    np.savetxt(filePath+"trans_1.txt",transformation_matrix)
    print("转换矩阵:\n", transformation_matrix)
    # 示例转换
    camera_point = np.array([0.09637157, 0.1824488, 0.67600006])  # 相机坐标系中的某个3D点
    #[0.09637157, 0.1824488, 0.67600006, 1] [-0.4125307   0.49836113  0.2646213   1.        ]
    #[0.09637157, 0.1824488, 0.67600006, 1] [-0.4125307   0.49836113  0.2646213   1.        ]
    #[0.09637157, 0.1824488, 0.67600006, 1] [-0.36713279  0.06529149  0.12877755]
    # 转换矩阵:
    #  [[-0.9804625  -0.18792206 -0.05812566 -0.19906497]
    #  [-0.18753625  0.80384455  0.56450341 -0.44490016]
    #  [-0.05935865  0.56437509 -0.82338163  0.58813451]
    #  [ 0.          0.          0.          1.        ]]
    # [-0.24627494 -0.00061246  0.1248572 ] 
    # [-0.29165235 0.43250328  0.26056038 ]
    robot_point = transform_point(camera_point, transformation_matrix)
    print("机械臂基座坐标系中的点:\n", robot_point)

def test_point():
    kinect = KinectCapture()
    tf = np.array([
        [-9.804624962067720606e-01, -1.879220621869180929e-01, -5.812565763414700992e-02, -1.990649741547012297e-01],
        [-1.875362468689325268e-01, 8.038445452634623845e-01, 5.645034128864878653e-01, -4.449001577233223093e-01],
        [-5.935865263212147108e-02,5.643750929954242102e-01, -8.233816276576163551e-01, 5.881345056000341076e-01],
        [0,0,0,1]
    ])
    edgePoints = np.column_stack((1208, 258))
    T_pinv = np.linalg.pinv(tf)
    result = kinect.search_3dImgIndex(edgePoints)
    print("the pos in kinect:")
    print(result)
    column_vector = np.array(result).T
    result = np.matmul(T_pinv, column_vector).T
    print("归一化之前")
    print(result)
    result_normalized = result[0][:3] / result[0][3]
    print("in rm65 归一化之后")
    print(result_normalized)


if __name__ == '__main__':
    print("---------------------")
    # lc_eye_cal() --暂时有用
    # exit()
    # 示例调用
    filename = 'data\\points\\6-28\\judge\\kinect_chess_3d.txt'  # 确保data.txt文件存在且格式正确
    # plot_3d_from_file(filename)
    # exit()
    test_point()
    exit()

    readPath = "data\\points\\6-28\\judge\\"
    # # 判断棋盘格角点顺序函数
    # img_path = readPath+'result.png'
    # size = (5, 5)  # Adjust this according to your chessboard size
    # corners, annotated_img = calbration_tools.find_and_display_chessboard_corners(img_path, size)
    # cv.imwrite(readPath+'annotated_chessboard_3.jpg', annotated_img)# Save the annotated image to visualize the corner points
    # # If you want to display the image
    # exit(0)

    chess_camera_pts, chess_color_pts_xy, chess_img, chess_corners = calbration_tools.read_data(
        readPath, "result")
    # np.savetxt(readPath + "\\chess_corners2d.txt", chess_corners)
    # 看了下这个顺序和利用kinect找到的相机角点顺序是一致的。
    # 目前的问题是利用该2d点找到的3d kinect_data数据顺序和机械臂好像不一样？明天再看看。
    #这里将棋盘格平面切割出来，chess_3d是整个棋盘格 3d信息，chess_2d包含其像素点的位置信息 
    chess_3d, chess_2d = calbration_tools.select_chessboard_pointcloud(img=chess_img, cameraPts=chess_camera_pts,
                                                        color_pts_xy=chess_color_pts_xy,
                                                        corners=chess_corners)
    kinect_chess_data = calbration_tools.homography_trans(chess_3d, chess_2d, chess_corners)
    # print(kinect_chess_data)
    # 这里通过一系列的方法先拟合棋盘格平面点云，划区域，找点，等等来确定25个点对应的位置信息，和本地直接通过kdtree找到的点云数据有一定的出入
    pts_kinect_chess = kinect_chess_data[:, :3]

    np.savetxt(readPath + "\\kinect_chess_3d.txt", pts_kinect_chess)

    robot_chess = np.loadtxt(readPath + "\\robot_Pts.txt")
    pts_chess_robot = robot_chess[:, :3]
    rotate_mat, trans, FRE = calbration_tools.caculate_conversion_matrix(pts_kinect_chess, pts_chess_robot)
    
    # test the reverse data
    # transposed_pts_chess_robot = pts_chess_robot[::-1]
    # rotate_mat, trans, FRE = calbration_tools.caculate_conversion_matrix(pts_kinect_chess, transposed_pts_chess_robot)

    # for test local data 
    # readPath = 'D:\\Program Files\\Company\\Jinjia\\Projects\\kinect\\data\\IO\\cacalbration_toolsration\\desk'
    # kinect_data = np.loadtxt(readPath + "\\kinect_corners_3d.txt")
    # robot_data = np.loadtxt(readPath + "\\desk_robot_Pts.txt")
    # rotate_mat, trans, FRE = calbration_tools.caculate_conversion_matrix(kinect_data, robot_data)
    
    print("R = ", np.array(rotate_mat))
    print("T= ", np.array(trans))
    print("FRE", FRE)
    T = np.concatenate([np.array(rotate_mat).transpose(), np.array(trans)], axis=0).transpose()
    np.savetxt(readPath + "\\transform_matrix.txt", T)
    # column_vector = np.array([[-32.5039], [ -58.7463], [616], [1]])

    # test_data = column_vector
    # result = np.matmul(T, test_data)
    # print(result/1000)

    # T = [
    #     [7.256057292488993227e-01, -5.899991142459756199e-01, -3.541149119567501558e-01, 4.671069980020330945e+02],
    #     [-6.879575926763845661e-01, -6.111545012448396097e-01 ,-3.914134978345600313e-01, 5.193349503707474923e+02],
    #     [1.451469462602163196e-02, 5.276279188946841892e-01, -8.493515778773698122e-01, 5.750616593360897468e+02]
    #     ]
    # column_vector = np.array([[-0.126147], [-0.0156571], [0.648], [1]])
    # result = np.matmul(T, column_vector)
    # print(result)

    #测试点数据
    # T = np.array([
    #     [7.256057292488993227e-01, -5.899991142459756199e-01, -3.541149119567501558e-01, 4.671069980020330945e+02],
    #     [-6.879575926763845661e-01, -6.111545012448396097e-01, -3.914134978345600313e-01, 5.193349503707474923e+02],
    #     [1.451469462602163196e-02, 5.276279188946841892e-01, -8.493515778773698122e-01, 5.750616593360897468e+02]
    # ])
    # column_vector = np.array([[-0.126147], [-0.0156571], [0.648], [1]])
    # T_pinv = np.linalg.pinv(T)
    # result = np.matmul(column_vector, T_pinv)
    # result = np.matmul(T, column_vector)
    # print(result)