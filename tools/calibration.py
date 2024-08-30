import numpy as np
import os
import sys
import time
import cv2 as cv
# 当前文件运行，也可以用python -m tools.calibration 这样运行，将当前文件作为包

# 从mian方法中调用
from .getKinect import KinectCapture
from .calbration_tools import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_robot_data(filePath):  # 传入文件路径后将文件加载并打印
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
        tmp = sorted(x, key=lambda i  : i[1])  
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

# 将传入的第一个参数points进行平移矩阵和旋转矩阵的转换，保存到savepath里
def kinect2robot_box(robot_pts, rotate_mat, trans, savePath):
    transformed_pts = np.matmul((robot_pts-trans.reshape([1, 3])), np.linalg.inv(rotate_mat.T))
    np.savetxt(savePath + "\\box_transformed_pts.txt", transformed_pts)
    return transformed_pts
  
def kinect2robot_desk(robot_pts, rotate_mat, trans, savePath):
    transformed_pts = np.matmul((robot_pts-trans.reshape([1, 3])), np.linalg.inv(rotate_mat.T))
    np.savetxt(savePath + "/desk_transformed_pts.txt", transformed_pts)
    return transformed_pts

def record_chess_order(readPath):
    # readPath = "data\\points\\7-1\\third\\"
    chess_camera_pts,chess_color_pts_xy, chess_img, chess_corners ,copy_img= read_data(readPath, "result")
    """
    10000*3,10000*2,相机拍摄的原图像，25*2（经25*1*2squeeze得来），标注corners后的图像
    """
    # 看了下这个顺序和利用kinect找到的相机角点顺序是一致的。
    #这里将棋盘格平面切割出来，chess_3d是整个棋盘格 3d信息，chess_2d包含其像素点的位置信息
    # 现在添加方法，在找棋盘格位置时，将顺序同步标记并保存，然后根据顺序完成机械臂点的对应标记点记录 
    chess_3d, chess_2d = select_chessboard_pointcloud(img=chess_img, cameraPts=chess_camera_pts,
                                                        color_pts_xy=chess_color_pts_xy,
                                                        corners=chess_corners)
    # 在方框内找的黑色像素点，1000*3，1000*2
    print(len(chess_3d))
    print(len(chess_2d))
    kinect_chess_data = homography_trans(chess_3d, chess_2d, chess_corners)
    # 这里通过一系列的方法先拟合棋盘格平面点云，划区域，找点，等等来确定25个点对应的位置信息，和本地直接通过kdtree找到的点云数据有一定的出入
    pts_kinect_chess = kinect_chess_data[:, :3]

    np.savetxt(readPath + "\\kinect_chess_3d.txt", pts_kinect_chess)
    if not os.path.exists(os.path.join(readPath,'robot_Pts.txt')):
        np.savetxt(readPath + "robot_Pts.txt",[])
    cv.imwrite(readPath+"\\show_corners_order.png",copy_img)

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

def save_trans_martix(filePath):
    # filePath = "data\\points\\7-1\\third\\"
    camera_points = read_points(filePath+"kinect_chess_3d.txt")
    robot_points = read_points(filePath+"robot_Pts.txt")
    robot_points /=1000
    # 计算转换矩阵
    transformation_matrix = hand_eye_calibration(camera_points, robot_points)
    np.savetxt(filePath+"trans_new.txt",transformation_matrix)
    print("转换矩阵:\n", transformation_matrix)

    # # 高师兄计算转换矩阵的方法
    # robot_chess = np.loadtxt(readPath + "\\robot_Pts.txt")
    pts_chess_robot = robot_points[:, :3]
    rotate_mat, trans, FRE = caculate_conversion_matrix(camera_points, pts_chess_robot)
    print("R = ", np.array(rotate_mat))
    print("T= ", np.array(trans))
    print("FRE", FRE)
    T = np.concatenate([np.array(rotate_mat).transpose(), np.array(trans)], axis=0).transpose()
    T_4x4 = np.eye(4)# 创建一个4x4的单位矩阵
    # T_4x4[:3, :3] = np.array(rotate_mat).transpose()# 将旋转矩阵的转置填入T_4x4的前3行前3列
    # T_4x4[:3, 3] = np.array(trans)# 将转换向量填入T_4x4的前3行第4列
    T_4x4[:3, :4] = T
    np.savetxt(readPath + "\\trans_old.txt", T_4x4)
    # 示例转换
    # camera_point = np.array([0.14206304, 0.10986008, 0.69200003])  # 相机坐标系中的某个3D点
    # robot_point = transform_point(camera_point, transformation_matrix)
    # print("机械臂基座坐标系中的点:\n", robot_point)

def cal_trans_data(edge_point):
    """
    计算相机空间与机械臂的转换对应关系
    
    参数:
    edge_point:相机空间一系列的二维点数组，也就是颜色空间的位置信息
    
    返回:
    numpy.ndarray: 机械臂基座坐标系中的3D点,形状为 (3,)。
    """
    
    
    # readPath= "data\\points\\7-24\\1st\\"
    readPath = "data\\points\\8-5\\calibrationAll\\"
    trans_martix = np.loadtxt(readPath+"trans_new.txt")
    kinect = KinectCapture()
    # edge_point = [[1262,362],[988,497]]
    data_in_kinect = kinect.search_3dImgIndex(edgePoints=edge_point)
    print("边缘点的个数为",len(data_in_kinect))
   
    print("边缘点的相机坐标为：")
    # print(data_in_kinect)
    result = []
    for i in data_in_kinect:
        data_in_rm = transform_point(i,trans_martix)
        # print(f"data in kinct:{i}")
        # print(f"data in rm65:{data_in_rm}")
        result.append(data_in_rm)
    print(f"the size of pointd in rm65: {len(result)}")
    return result

# cyl
def transform_rm_65(point,transformation_matrix):
    point_homogeneous = np.append(point,1)
    transformed_point_homogeneous = np.dot(np.linalg.inv(transformation_matrix),point_homogeneous)
    transformed_point = transformed_point_homogeneous[:3] * transformed_point_homogeneous[3]
    return transformed_point
'''
# def data_trans_cal(rm_65_point)
def data_trans_cal(readPath):
    rm_65_point = 'data/points/08-14/first/plan_data_update.txt'
    kinectCapture = KinectCapture()

    trans_matrix = np.loadtxt(readPath+'trans_new.txt')
    result = []
    with open(rm_65_point,'r') as file:
        lines = file.readlines()
        for l in lines:
            points_xyz = np.array([float(val) for val in l.strip().split()[:3]])
            data_in_depth = transform_rm_65(points_xyz,trans_matrix)
            result.append(data_in_depth.tolist())
    print(result)

# cyl
'''
def data_trans_cal(readPath_matrix,read_path_point):
    # rm_65_point = readPath+'plan_data.txt'
    # rm_65_point = read_path_point + 'first' + 'plan_data_update.txt'
    rm_65_point = os.path.join(read_path_point,'plan_data.txt')
    
    
    '''
    kinectCapture = KinectCapture()
    color_image,depth_img = kinectCapture.get_frames()
    depth_frame = kinectCapture.kinect.get_last_depth_frame()
    # 获取相机空间点
    camera_space_points = kinectCapture.get_camera_space_points(depth_frame)
    # 获取颜色空间点
    color_space_points = kinectCapture.get_color_space_points(depth_frame)

    filter_pts_3d,filter_pts_2d = [],[]

    for i,point in enumerate(camera_space_points):
        x,y,z = float(point.x),float(point.y),float(point.z)
        if not (np.isinf(x) or np.isinf(y) or np.isinf(z)):  # 如果点的坐标不是inf或-inf
                color_space_point = color_space_points[i]
                x_c, y_c = float(color_space_point.x), float(color_space_point.y)
                if 0 <= x_c < color_image.shape[1] and 0 <= y_c < color_image.shape[0]:
                    color = [x_c,y_c]
                    point = [point.x, point.y, point.z]
                    filter_pts_3d.append(point)
                    filter_pts_2d.append(color)
    filter_pts_3d,filter_pts_2d = np.array(filter_pts_3d),np.array(filter_pts_2d)
    '''
    filter_pts_3d,filter_pts_2d = read_data_cyl(read_path_point, "result")
    # print('filter_pts_2d.shape',filter_pts_2d.shape)
    # print('filter_pts_3d.shape',filter_pts_3d.shape)

    

    trans_matrix = np.loadtxt(readPath_matrix+'trans_new.txt')
    result = []
    with open(rm_65_point,'r') as file:
        lines = file.readlines()
        for l in lines:
            points_xyz = np.array([float(val) for val in l.strip().split()[:3]])
            data_in_depth = transform_rm_65(points_xyz,trans_matrix)
            result.append(data_in_depth)
        result = np.array(result)
    
    adjusted_pts,points_center_all = plane_fitting(filter_pts_3d,"_")
    adjusted_pts_less = plane_fitting(result,points_center_all)
    eigen_vals,eigen_mat = pca(adjusted_pts)
    min_pos = eigen_vals.argmin()

    pca_res = np.matmul(adjusted_pts,eigen_mat)
    pca_res_less = np.matmul(adjusted_pts_less,eigen_mat)
    pca_res = np.delete(pca_res,[min_pos],axis=1)
    pca_res_less = np.delete(pca_res_less,[min_pos],axis=1)

    homography_mat,_ = cv.findHomography(filter_pts_2d,pca_res)
    # 这行代码使用perspectiveTransform函数对filtered_pts_2d进行透视变换，使用homography_mat作为变换矩阵，得到变换后的点坐标数组warped_all。
    warped_all = cv.perspectiveTransform(np.expand_dims(filter_pts_2d, axis=0), homography_mat)[0]

    error = (warped_all - pca_res) * 1e3
    # print('warped_all rms:', get_rms(error))

    homography_mat_inv = np.linalg.inv(homography_mat)
    warped_res = cv.perspectiveTransform(np.expand_dims(pca_res_less,axis=0),homography_mat_inv)
    warped_res = np.squeeze(warped_res)
    # print(warped_res)
    return warped_res


if __name__ == '__main__':
    from getKinect import KinectCapture
    from calbration_tools import *
    print("---------------------")
    readPath = "data\\points\\8-5\\calibrationRed\\"

    t = time.strftime('%m-%d', time.localtime())
    file_index = "first"
    run_record_path = "data\\points\\"+t+"\\"+file_index+"\\"
    1 # 保存点云信息
    # os.makedirs(os.path.join(readPath), exist_ok=True)
    # kinect = KinectCapture()
    # kinect.save_point_cloud(filename=readPath)
    # sys.exit()
    2
    record_chess_order(readPath)#首先运行该函数找到棋盘格角点的顺序及坐标信息
    exit()
    # # 记录机械臂在这些点的位置信息得到robot_Pts.txt文档，记录顺序与角点顺序一致
    # 3
    # save_trans_martix(readPath) #然后运行该方法得到其转换矩阵
    # exit()
    # 4--测试
    edge_point = [[760.03648706,349.95831667],[780.59977034,348.54209752],
                  [801.35294001,347.29199446],[822.22347011,346.69202168],
                  [843.29738312,346.71451462],[864.32438176,346.83895831],[885.40052939,346.92360397]]
    cal_trans_data(edge_point)# 传入像素数组，进行测试or计算--记得修改读取权重文件的路径
    data_trans_cal(readPath,run_record_path)

    exit()
    # # 示例调用
    plot_3d_from_file(readPath+"\\wound\\wound_data_rm65.txt")
    file_path = "data\\points\\7-8\\4th\\"
    camera_points_file = file_path + "camera_points.txt"
    robot_points_file = file_path + "robots_points.txt"
    transformation_matrix_file = file_path + "transf.txt"
