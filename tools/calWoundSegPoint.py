import lib
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def func(x, a, b, c, d, e, f, g):
    # 最小二乘法
    # return a*x**6 + b*x**5 + c*x**4 + d*x**3 + e*x**2 + f*x + g
    # 二阶多项式
    return a*x**2 + b*x + c

def your_fit_function(data, *params):
    x, y = data
    # 二次多项式拟合函数示例：
    # z = a + bx + cy + dx^2 + ey^2 + fxy
    a, b, c, d, e, f = params
    return a + b * x + c * y + d * x**2 + e * y**2 + f * x * y


# def func(x, y, z, *params):
#     return your_fit_function(x, y, z, *params)

def plot_wound(file_path,num_segments):
    data = np.loadtxt(file_path+'wound_3d_RM65_shape.txt')
    
    x = data[:, 0]
    y = data[:, 1]
    # z = data[:, 2]
    temp_z = 59.5
    # 调用curve_fit函数进行拟合
    popt, pcov = curve_fit(func, x, y)

    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min

    # 计算每个分段的长度
    segment_length = x_range / num_segments

    # 初始化保存中间值的数组
    segment_points = np.zeros((num_segments, 3))
    segment_points_download = np.zeros((num_segments, 3))
    # 遍历每个分段
    for i in range(num_segments):
        # 计算当前分段的起始和结束 x 值
        start_x = x_min + i * segment_length
        end_x = x_min + (i + 1) * segment_length

        # 找到当前分段对应的数据索引
        segment_indices = np.where((x >= start_x) & (x < end_x))[0]

        # 计算当前分段的中间值
        segment_x = np.mean(x[segment_indices])
        segment_y = func(segment_x, *popt)
        segment_points[i] = [segment_x, segment_y, temp_z]
        segment_points_download[i] = [segment_x+35, segment_y-7, temp_z]
    # 保存中间值到本地 txt 文档
    np.savetxt(file_path + "segment_points_shape.txt", segment_points_download[segment_points_download[:, 0].argsort()], fmt='%.6f')
    # 提取x和y坐标
    x_curve = segment_points[:, 0]
    y_curve = segment_points[:, 1]

    # 绘制散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, c='r', label='Original Point')
    ax.scatter(x_curve, y_curve, c='b', label='Segement Point')

    # 绘制拟合曲线
    x_curve = np.linspace(min(x), max(x), 1000)
    y_curve = func(x_curve, *popt)
    plt.plot(x_curve, y_curve, 'g-', label='Polynomial Fit')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title('Segment Points Visualization')
    plt.legend()
    plt.show()


def nihe3d(file_path,num_segments):
    data = np.loadtxt(file_path+'wound_3d_RM65_shape.txt')
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    # temp_z = 155
    # 调用curve_fit函数进行拟合
    params_guess = [1, 2, 3, 4, 5, 6]  # 初始参数值列表，数量为 6
    popt, pcov = curve_fit(your_fit_function, (x, y), z, p0=params_guess)
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min

    # 计算每个分段的长度
    segment_length = x_range / num_segments

    # 初始化保存中间值的数组
    segment_points = np.zeros((num_segments, 3))

    # 遍历每个分段
    for i in range(num_segments):
        # 计算当前分段的起始和结束 x 值
        start_x = x_min + i * segment_length
        end_x = x_min + (i + 1) * segment_length

        # 找到当前分段对应的数据索引
        segment_indices = np.where((x >= start_x) & (x < end_x))[0]

        # 计算当前分段的中间值
        segment_x = np.mean(x[segment_indices])
        segment_y = np.mean(y[segment_indices])
        segment_z = func(segment_x, segment_y, *popt)
        segment_points[i] = [segment_x, segment_y, segment_z]
    # 保存中间值到本地 txt 文档
    np.savetxt(file_path + "segment_points_shape.txt", segment_points[segment_points[:, 0].argsort()], fmt='%.6f')
    # 提取x和y坐标
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', label='Original Point')
    ax.scatter(x_curve, y_curve, z_curve, c='b', label='Segement Point')

    # 绘制拟合曲线
    x_curve = np.linspace(min(x), max(x), 100)
    y_curve = np.linspace(min(y), max(y), 100)
    X, Y = np.meshgrid(x_curve, y_curve)
    Z = func(X, Y, *popt)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, label='3D Fit')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Segment Points Visualization')
    plt.legend()
    plt.show()

def nihe(file_path):
    # 加载二维点数据
    data = np.loadtxt(file_path+'wound_3d_RM65_shape.txt')

    # 提取x和y坐标
    x = data[:, 0]
    y = data[:, 1]

    # 进行9次多项式拟合
    coefficients = np.polyfit(x, y, 6)
    polynomial = np.poly1d(coefficients)

    # 计算分割点
    num_segments = 60
    segment_points = np.linspace(min(x), max(x), num_segments)

    # 计算每个分割点的值
    segment_values = polynomial(segment_points)

    # 绘制原始数据点和分割点
    plt.scatter(x, y, c='r', label='Original Data')
    plt.scatter(segment_points, segment_values, c='b', label='Segment Points')
    # 保存中间节点
    # np.savetxt(file_path+"segment_points_shape.txt", segment_points[segment_points[:, 1].argsort()[::-1]], fmt='%.6f')
    # 绘制拟合曲线
    x_curve = np.linspace(min(x), max(x), 1000)
    y_curve = polynomial(x_curve)
    plt.plot(x_curve, y_curve, 'g-', label='Polynomial Fit')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polynomial Fit and Segment Points')
    plt.legend()
    plt.show()

def plot_3d_points(file_path):
    # 从文本文件中读取三维点
    data = np.loadtxt(file_path+"wound_3d_Kinect_shape.txt")

    # 根据 Z 轴的值对数据进行排序
    sorted_data = data[data[:, 2].argsort()]

    # 选择除了最小的两个点之外的点
    # filtered_data = sorted_data[2:, :]
    # 平面拟合看看
    # data = lib.plane_fitting(data)

    # 将三维点的坐标分离
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # 计算 Z 值的和与平均值
    z_sum = np.sum(z)
    z_mean = np.mean(z)

    # 创建一个三维坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置纵横比
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    z_range = np.max(z) - np.min(z)
    max_range = max(x_range, y_range, z_range)
    ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])

    # 显示 Z 值的和与平均值
    ax.text2D(0.05, 0.90, f"Mean of Z: {z_mean:.2f}", transform=ax.transAxes)

    # 显示图形
    plt.show()

def get_z_RM65(file_path):
     # 从文本文件中读取三维点
    data = np.loadtxt(file_path)
    # 根据 Z 轴的值对数据进行排序
    sorted_data = data[data[:, 2].argsort()]
    # 选择除了最小的两个点之外的点
    filtered_data = sorted_data[2:, :]
    # 将三维点的坐标分离
    x = filtered_data[:, 0]
    y = filtered_data[:, 1]
    z = filtered_data[:, 2]
    # 计算 Z 值的和与平均值
    z_sum = np.sum(z)
    z_mean = np.mean(z)
    # 显示图形
    return z_mean

def estimate_similarity_transformation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Estimate similarity transformation (rotation, scale, translation) from source to target (such as the Sim3 group).
    """
    k, n = source.shape

    mx = source.mean(axis=1)
    my = target.mean(axis=1)
    source_centered = source - np.tile(mx, (n, 1)).T
    target_centered = target - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(source_centered**2, axis=0))
    sy = np.mean(np.sum(target_centered**2, axis=0))

    Sxy = (target_centered @ source_centered.T) / n

    U, D, Vt = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = Vt.T
    rank = np.linalg.matrix_rank(Sxy)
    if rank < k:
        raise ValueError("Failed to estimate similarity transformation")

    S = np.eye(k)
    if np.linalg.det(Sxy) < 0:
        S[k - 1, k - 1] = -1

    R = U @ S @ V.T

    s = np.trace(np.diag(D) @ S) / sx
    t = my - s * (R @ mx)

    return R, s, t

def zhuanzhi(data):
    camera_data = np.array(data)
    # 打印二维数组
    camera_data = camera_data.astype(float)
    return camera_data

def calibration(filePath):
    # 读取包含棋盘格的图片
    img = cv2.imread(filePath)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取棋盘角点
    ret, corners = cv2.findChessboardCorners(img, (5, 5), None)

    if ret:
        # 将corners转换为numpy.ndarray类型
        corners = np.int0(corners)

        # 标注角点并显示顺序号
        for i, corner in enumerate(corners):
            cv2.circle(img, (corner[0][0], corner[0][1]), 5, (0, 255, 0), -1)
            cv2.putText(img, str(i+1), (corner[0][0]+10, corner[0][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 输出角点信息
        print('棋盘角点像素位置信息：')
        for i, corner in enumerate(corners):
            print(f'{i+1}. ({corner[0][0]}, {corner[0][1]})')

        # 保存角点信息到本地txt文件
        # with open('chessboard_corners.txt', 'w') as f:
        #     for i, corner in enumerate(corners):
        #         f.write(f'{i+1},{corner[0][0]},{corner[0][1]}\n')
        # 保存标注后的棋盘图像到本地
        cv2.imwrite('chessboard_annotated.jpg', img)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('Failed to detect chessboard corners in the image.')

def findCorners(file_path):
    # 加载棋盘格图像
    image = cv2.imread(file_path)

    # 将图像转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 设置棋盘的尺寸
    pattern_size = (5, 5)  # 这是你的棋盘格的角点数量，可能需要调整

    # 查找棋盘角点
    found, corners = cv2.findChessboardCorners(gray, pattern_size)

    if found:
        # 优化角点检测
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)

        # 在原图中标记检测到的角点
        cv2.drawChessboardCorners(image, pattern_size, corners, found)

        # 显示图像
        cv2.imshow('Chessboard Corners', image)
        cv2.waitKey()

    cv2.destroyAllWindows()

def plot_RM_65(file_path):
    data = np.loadtxt(file_path+'segment_points_shape.txt')
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    new_z = []

    plot_data = np.zeros((data.size, 3))

    # 遍历每个分段
    for i in range(len(z)):
        new_z.append((20 - i * 2.20)+z[i])
    # 转换为NumPy数组
    new_z = np.array(new_z)
    # 创建一个空数组来存储折线连接点的x、y和z坐标
    connected_x = []
    connected_y = []
    connected_z = []
    for i in range(len(z) - 1):
        connected_x.extend([x[i], x[i], x[i + 1]])
        connected_y.extend([y[i], y[i], y[i + 1]])
        connected_z.extend([z[i], new_z[i], z[i + 1]])

    # 转换为NumPy数组
    connected_x = np.array(connected_x)
    connected_y = np.array(connected_y)
    connected_z = np.array(connected_z)


    # 创建一个三维坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置纵横比
    x_range = np.max(connected_x) - np.min(connected_x)
    y_range = np.max(connected_y) - np.min(connected_y)
    z_range = np.max(connected_z) - np.min(connected_z)
    max_range = max(x_range, y_range, z_range)
    ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])

    ax.plot(connected_x, connected_y, connected_z,'-o', color='b', markersize=5)
    # 添加标题
    ax.set_title("Schematic diagram of the actual movement trajectory of the robotic arm")

    plt.show()


def plot_point_test(file_path):
    # 从文本文件中读取三维点
    data = np.loadtxt(file_path)

    # 根据 Z 轴的值对数据进行排序
    sorted_data = data[data[:, 2].argsort()]

    # 选择除了最小的两个点之外的点
    # filtered_data = sorted_data[2:, :]
    # 平面拟合看看
    # data = lib.plane_fitting(data)

    # 将三维点的坐标分离
    x = data[:, 0]
    y = data[:, 1]
    # z = data[:, 2]
    z = 0
    # 计算 Z 值的和与平均值
    z_sum = np.sum(z)
    z_mean = np.mean(z)

    # 创建一个三维坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置纵横比
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    z_range = np.max(z) - np.min(z)
    max_range = max(x_range, y_range, z_range)
    ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])

    # 显示 Z 值的和与平均值
    ax.text2D(0.05, 0.90, f"Mean of Z: {z_mean:.2f}", transform=ax.transAxes)

    # 显示图形
    plt.show()

if __name__ == '__main__':
    print("---------------------")
    # 伤口显示3d 
    # plot_3d_points('D:\\Program Files\\Company\\Jinjia\\Projects\\programs\\testWound\\data\\')
    
    # plot_wound('D:\\Program Files\\Company\\Jinjia\\Projects\\kinect\\data\\beiGongda\\newChess\\newCalibration\\testPoint\\',12)
    
    #轨迹规划 
    # plot_wound('D:\\Program Files\\Company\\Jinjia\\Projects\\programs\\testWound\\data\\',13)
    
    # 机械臂末端轨迹显示
    # plot_RM_65('D:\\Program Files\\Company\\Jinjia\\Projects\\programs\\testWound\\')

    #
    filePath = "D:\\Program Files\\Company\\Jinjia\\Projects\\auotoSuture\\auotoSutureWound\\algorithm_C++\\data\\"
    # wound_3d_edge_inKinect.txt wound_3d_RM65_shape wound_3d_RM65_edge
    plot_point_test(filePath+"wound_3d_RM65_shape.txt")
