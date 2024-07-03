# coding=utf-8
import os
import cv2 as cv
import numpy as np
import open3d as o3d
print(cv.__file__)


def find_chessboard(img_path, size):
    assert os.path.exists(img_path)
    copyImg = cv.imread(img_path)
    ok, corners_first = cv.findChessboardCorners(copyImg, size, None)

    font = cv.FONT_HERSHEY_SIMPLEX
    if ok:
        point0 = (int(np.squeeze(corners_first[0])[0]), int(np.squeeze(corners_first[0])[1]))
        point1 = (int(np.squeeze(corners_first[24])[0]), int(np.squeeze(corners_first[24])[1]))
        cv.rectangle(copyImg, point0, point1, (0, 0, 255), -1)
        ok, corners_sec = cv.findChessboardCorners(copyImg, size, None)
        if ok:
            point0 = (int(np.squeeze(corners_sec[0])[0]), int(np.squeeze(corners_sec[0])[1]))
            point1 = (int(np.squeeze(corners_sec[24])[0]), int(np.squeeze(corners_sec[24])[1]))
            cv.rectangle(copyImg, point0, point1, (0, 0, 255), -1)
        else:
            # 当摆放角度与现有坐标系不平行时，这样分隔开的矩形区域就找不到所有的角点，所有这里先直接返回~
            return np.squeeze(corners_first)
        #这里将两次找到的点位位置进行求和平均，而不是直接用concatenate()进行合并计算？
        # corners = np.concatenate([corners_first, corners_sec], 0)
        merged_martix = np.add(corners_first,corners_sec)
        corners = np.divide(merged_martix,2)

        corners = np.squeeze(corners)
        return corners
    else:
        print('cannot find chessboard points')
        return -1


def find_and_display_chessboard_corners(img_path, size):
    # Ensure the image path exists
    assert os.path.exists(img_path), "Image path does not exist."
    # Read the image
    copyImg = cv.imread(img_path)
    if copyImg is None:
        raise ValueError("Could not read the image.")
    # Find chessboard corners
    ok, corners_first = cv.findChessboardCorners(copyImg, size, None)
    font = cv.FONT_HERSHEY_SIMPLEX
    if ok:
        # Draw and label the first set of corners
        for i, corner in enumerate(corners_first):
            point = (int(corner[0][0]), int(corner[0][1]))
            cv.circle(copyImg, point, 5, (0, 255, 0), -1)  # Draw a circle at each corner BGR的顺序
            cv.putText(copyImg, str(i), point, font, 0.5, (255, 0, 0), 1, cv.LINE_AA)  # Put the index number
        # Modify the image by drawing a rectangle around the first and last corner of the first detection
        point0 = (int(corners_first[0][0][0]), int(corners_first[0][0][1]))
        point1 = (int(corners_first[24][0][0]), int(corners_first[24][0][1]))
        # cv.rectangle(copyImg, point0, point1, (0, 0, 255), -1)
        ok, corners_sec = cv.findChessboardCorners(copyImg, size, None)
        if ok: 
            for i, corner in enumerate(corners_sec):# Draw and label the second set of corners
                point = (int(corner[0][0]), int(corner[0][1]))
            # Average the positions of the two sets of corners
            merged_matrix = np.add(corners_first, corners_sec)
            corners = np.divide(merged_matrix, 2)
            corners = np.squeeze(corners)
            return corners, copyImg
        else:
            print("Cannot find chessboard points in the modified image.")
            return np.squeeze(corners_first), copyImg
    else:
        print('Cannot find chessboard points.')
        return -1, copyImg


def pca(point_set):
    covariance_mat = np.cov(point_set, rowvar=False)
    eigen_vals, eigen_vecs = np.linalg.eig(covariance_mat)

    return eigen_vals, eigen_vecs


def get_rms(array):
    a = array * array
    rms = np.sum(a, -1)
    rms = np.mean(rms, -1)
    rms = np.sqrt(rms)
    return rms


def get_plane_info(point, vector):
    vector /= np.linalg.norm(vector)  # 标准化
    A, B, C = vector[0], vector[1], vector[2]

    D = -point.dot(vector)
    return A, B, C, D


def proj_pts2plane(src_pts, plane_info):
    A, B, C, D = plane_info
    vec = np.array([A, B, C])

    res = np.zeros(src_pts.shape)
    for i in range(src_pts.shape[0]):
        src_pt = src_pts[i]

        k = -vec.dot(src_pt) - D
        x, y, z = src_pt + k * vec
        res[i] = np.array([x, y, z])

    return res


def plane_fitting(point_set):
    eigen_vals, eigen_vecs = pca(point_set)

    min_eigen_val_index = eigen_vals.argmin()  # np.argmin()查找最小值，并返回其下标
    vec = eigen_vecs[::, min_eigen_val_index]  # vec为最小特征值对应的特征向量，用于代表平面的法向量

    pts_center = np.mean(point_set, axis=0)
    plane_coef = get_plane_info(pts_center, vec)

    pts_proj = proj_pts2plane(point_set, plane_coef)

    return pts_proj


def pts_coplane(pts):
    eigen_vals, eigen_vecs = pca(pts)

    if eigen_vals.min(-1) < 1e-5:
        return True

    return False


def rigid_transform_3D(A, B):
    assert len(A) == len(B) , "the len(A)!=len(B)"

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # centre the points
    AA = A - A.mean(axis=0)
    BB = B - B.mean(axis=0)

    H = np.matmul(np.transpose(AA), BB)
    U, S, Vt = np.linalg.svd(H)

    R = np.matmul(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.matmul(Vt.T, U.T)

    t = -np.matmul(R, centroid_A) + centroid_B
    return R, t.reshape((1, 3))


# 计算均方根（反应的时有效值）
def calculate_rms(array):
    a = array * array
    rms = np.sum(a, -1)
    rms = np.mean(rms, -1)
    rms = np.sqrt(rms)
    return rms

#保存为点云文件
def save_point_cloud_to_pcd(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")


# **FLE:**即定位基准点的误差
# **FRE:**配准后相应基准点之间的均方根距离
# **TRE:**配准后基准点以外的相应点之间的距离

def select_chessboard_pointcloud(img, cameraPts, color_pts_xy, corners):
    assert len(cameraPts) == len(color_pts_xy)
    x_min = np.min(corners[:, 0])
    x_max = np.max(corners[:, 0])
    y_min = np.min(corners[:, 1])
    y_max = np.max(corners[:, 1])

    pool = []
    for index, pt in enumerate(color_pts_xy):
        x, y = pt
        if (x_min < x < x_max) and (y_min < y < y_max):
            rgb = img[int(y), int(x)]
            if min(rgb) >= 128:
                pool.append(index)

    selected_pts_3d = cameraPts[pool, :]
    selected_pts_2d = color_pts_xy[pool, :]

    return selected_pts_3d, selected_pts_2d


def homography_trans(filtered_pts_3d, filtered_pts_2d, filtered_corners):
    # save_point_cloud_to_pcd(filtered_pts_3d, "origin_point_cloud.pcd")
    adjusted_pts = plane_fitting(filtered_pts_3d)

    # save_point_cloud_to_pcd(adjusted_pts, "adjusted_point_cloud.pcd")

    eigen_vals, eigen_mat = pca(adjusted_pts)
    min_pos = eigen_vals.argmin()
    # 数据在特征向量空间中的投影结果 
    pca_res = np.matmul(adjusted_pts, eigen_mat)
    # 这行代码将pca_res中第一行、第min_pos列的元素赋值给变量sta。
    sta = pca_res[0][min_pos]
    # 这行代码使用np.delete函数删除pca_res中的第min_pos列，结果存储在pca_res中
    pca_res = np.delete(pca_res, [min_pos], axis=1)
    # 这行代码调用了OpenCV的findHomography函数，根据filtered_pts_2d和pca_res计算出一个单应性矩阵homography_mat，用于将二维点坐标映射到特征向量空间。 
    homography_mat, _ = cv.findHomography(filtered_pts_2d, pca_res)
    # 这行代码使用perspectiveTransform函数对filtered_pts_2d进行透视变换，使用homography_mat作为变换矩阵，得到变换后的点坐标数组warped_all。
    warped_all = cv.perspectiveTransform(np.expand_dims(filtered_pts_2d, 0), homography_mat)[0]
    error = (warped_all - pca_res) * 1e3
    print('warped_all rms:', get_rms(error))
    # 下两行代码对filtered_corners进行透视变换，使用homography_mat作为变换矩阵，得到变换后的角点坐标数组warped_res
    print(filtered_corners)
    warped_res = cv.perspectiveTransform(np.expand_dims(filtered_corners, 0), homography_mat)
    warped_res = np.squeeze(warped_res)
    print(warped_res)
    # 这行代码在warped_res的第min_pos列插入值为sta的元素，使其与原始投影结果的维度保持一致。
    warped_res = np.insert(warped_res, min_pos, values=[sta] * len(warped_res), axis=1)
    # 这部分代码通过线性组合计算得到结果数组res。对于warped_res中的循环中的每一行，获取x、y、z坐标，并使用特征向量矩阵eigen_mat按权重进行线性组合，将结果存储在res中
    res = np.zeros(warped_res.shape)
    for i in range(warped_res.shape[0]):
        x, y, z = warped_res[i]
        res[i, :] = x * eigen_mat[:, 0] + y * eigen_mat[:, 1] + z * eigen_mat[:, 2]
    return res


# **TRE:**配准后基准点以外的相应点之间的距离
def caculate_tre(remain_kinect_pts, remain_robot_pts, rotate_mat, trans):
    tre = caculate_fre(remain_kinect_pts, remain_robot_pts, rotate_mat, trans)
    return tre


#  **FRE:**配准后相应基准点之间的均方根距离
def caculate_fre(kinect_pts, robot_pts, rotate_mat, trans):
    transformed_pts = np.matmul(kinect_pts, rotate_mat.T) + trans.reshape([1, 3])
    fre = get_rms((transformed_pts - robot_pts))
    return fre


def caculate_conversion_matrix(kinect_chess_data, pts_chess_robot):
    kinect_pts = kinect_chess_data
    robot_pts = pts_chess_robot
    r, t = rigid_transform_3D(kinect_pts, robot_pts)
    fre = caculate_fre(kinect_pts, robot_pts, r, t)
    print(r, t, fre)

    return r, t, fre


def read_data(filePath, fileNmae):
    camera_space_path = filePath + "\\" + fileNmae + "_cameraPts.txt"
    color_space_path = filePath + "\\" + fileNmae + "_colorPts.txt"
    img_path = filePath + "\\" + fileNmae + ".png"
    # a = os.path.dirname(os.path.realpath(sys.executable))
    # b = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
    # print(b)
    # corners = find_chessboard(img_path=img_path, size=(5, 5))
    corners,copy_img = find_and_display_chessboard_corners(img_path=img_path, size=(5, 5))
    camera_pts = np.loadtxt(camera_space_path)
    # print(camera_pts.shape)
    color_pts_xy = np.loadtxt(color_space_path)
    img = cv.imread(img_path)

    return camera_pts, color_pts_xy, img, corners,copy_img
