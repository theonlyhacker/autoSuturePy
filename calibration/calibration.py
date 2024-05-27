import lib
import numpy as np
import os
import sys

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


if __name__ == '__main__':
    print("---------------------")
    # 机械臂采集的角点坐标和kinect采集的角点坐标的txt文件保存路径
    # readPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(sys.executable))))
    # readPath = os.path.join(readPath,
    #                         r".\\Outdata")
    
    readPath = 'D:\\Program Files\\Company\\Jinjia\\Projects\\kinect\\data\\beiGongda\\newChess\\newCalibration'

    chess_camera_pts, chess_color_pts_xy, chess_img, chess_corners = lib.read_data(
        readPath, "desk")
    # np.savetxt(readPath + "\\chess_corners2d.txt", chess_corners)
    # 看了下这个顺序和利用kinect找到的相机角点顺序是一致的。
    # 目前的问题是利用该2d点找到的3d kinect_data数据顺序和机械臂好像不一样？明天再看看。
    #这里将棋盘格平面切割出来，chess_3d是整个棋盘格 3d信息，chess_2d包含其像素点的位置信息 
    chess_3d, chess_2d = lib.select_chessboard_pointcloud(img=chess_img, cameraPts=chess_camera_pts,
                                                        color_pts_xy=chess_color_pts_xy,
                                                        corners=chess_corners)
    kinect_chess_data = lib.homography_trans(chess_3d, chess_2d, chess_corners)
    # print(kinect_chess_data)
    # 这里通过一系列的方法先拟合棋盘格平面点云，划区域，找点，等等来确定25个点对应的位置信息，和本地直接通过kdtree找到的点云数据有一定的出入
    pts_kinect_chess = kinect_chess_data[:, :3]*1000

    np.savetxt(readPath + "\\kinect_chess_3d.txt", pts_kinect_chess)

    robot_chess = np.loadtxt(readPath + "\\robot_Pts.txt")
    pts_chess_robot = robot_chess[:, :3]
    rotate_mat, trans, FRE = lib.caculate_conversion_matrix(pts_kinect_chess, pts_chess_robot)
    
    # test the reverse data
    # transposed_pts_chess_robot = pts_chess_robot[::-1]
    # rotate_mat, trans, FRE = lib.caculate_conversion_matrix(pts_kinect_chess, transposed_pts_chess_robot)

    # for test local data 
    # readPath = 'D:\\Program Files\\Company\\Jinjia\\Projects\\kinect\\data\\IO\\calibration\\desk'
    # kinect_data = np.loadtxt(readPath + "\\kinect_corners_3d.txt")
    # robot_data = np.loadtxt(readPath + "\\desk_robot_Pts.txt")
    # rotate_mat, trans, FRE = lib.caculate_conversion_matrix(kinect_data, robot_data)
    
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