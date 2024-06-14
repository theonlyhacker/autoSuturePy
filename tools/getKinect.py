import cv2
import numpy as np
import time
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import pclpy
from pclpy import pcl
from sortedcontainers import SortedSet
import open3d as o3d
import ctypes
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

depthSize = 512*424

class KinectCapture:
    def __init__(self):
        # 初始化 Kinect
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
        # 等待3秒钟，直到相机准备好
        time.sleep(3)
        self._mapper = self.kinect._mapper
    
    # 添加一个判断相机是否连接成功的标志
    def connect(self):
        color, depth = self.get_frames()
        if color is not None and depth is not None:
            return True
        return False
    
    def get_frames(self):
        # 获取颜色帧和深度帧
        if self.kinect.has_new_color_frame() and self.kinect.has_new_depth_frame():
            color_frame = self.kinect.get_last_color_frame()
            depth_frame = self.kinect.get_last_depth_frame()
            # 将颜色帧转换为图像格式
            color_image = color_frame.reshape((self.kinect.color_frame_desc.Height, self.kinect.color_frame_desc.Width, 4), order='C')
            color_image = color_image[:, :, :3]  # 去掉 alpha 通道
            # 这里本来就是RGB格式，不需要转换   2024.2.29
            # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
            # 将深度帧转化为图像格式
            depth_image = depth_frame.reshape(self.kinect.depth_frame_desc.Height, self.kinect.depth_frame_desc.Width)
            depth_image = (depth_image).astype(np.uint8)  # 适当的缩放和映射
            return color_image, depth_image
        
        return None, None

    def get_camera_space_points(self, depth_frame):
        # 创建一个空的_CameraSpacePoint数组来存储结果
        camera_space_points = (PyKinectV2._CameraSpacePoint * depth_frame.size)()
        # 将深度帧数据转换为LP_c_ushort类型
        depth_frame_data = depth_frame.flatten().astype(np.uint16)
        depth_frame_data_p = depth_frame_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        # 调用MapDepthFrameToCameraSpace方法
        self._mapper.MapDepthFrameToCameraSpace(depth_frame.size, depth_frame_data_p, depth_frame.size, camera_space_points)
        # 打印深度帧对应的相机空间点
        # for point in camera_space_points:
        #     print("X: {}, Y: {}, Z: {}".format(point.x, point.y, point.z))
        return camera_space_points
    
    def get_color_space_points(self, depth_frame):
        # 创建一个空的_ColorSpacePoint数组来存储结果
        color_space_points = (PyKinectV2._ColorSpacePoint * depth_frame.size)()
        # 将深度帧数据转换为LP_c_ushort类型
        depth_frame_data = depth_frame.flatten().astype(np.uint16)
        depth_frame_data_p = depth_frame_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        # 调用MapDepthFrameToColorSpace方法
        self._mapper.MapDepthFrameToColorSpace(depth_frame.size, depth_frame_data_p, depth_frame.size, color_space_points)
        # 打印深度帧对应的颜色空间点
        # for point in color_space_points:
        #     print("X: {}, Y: {}".format(point.x, point.y))
        # print("Size of color_space_points: ", len(color_space_points))
        return color_space_points

    def get_point_cloud(self,color_image):
        # 创建点云
        point_cloud = o3d.geometry.PointCloud()
        # 获取深度帧 
        depth_frame = self.kinect.get_last_depth_frame()
        # color_image, depth_image = self.get_frames()
        # 获取相机空间点
        camera_space_points = self.get_camera_space_points(depth_frame)
        # 获取颜色空间点
        color_space_points = self.get_color_space_points(depth_frame)
        # 为每个深度点分配颜色
        for i, point in enumerate(camera_space_points):
            x, y, z = float(point.x), float(point.y), float(point.z)
            if not (np.isinf(x) or np.isinf(y) or np.isinf(z)):  # 如果点的坐标不是inf或-inf
                color_space_point = color_space_points[i]
                x_c, y_c = int(color_space_point.x), int(color_space_point.y)
                if 0 <= x_c < color_image.shape[1] and 0 <= y_c < color_image.shape[0]:
                    color = color_image[y_c, x_c]/255.0 # 归一化 open3d颜色在0-1之间
                    point = [point.x, point.y, point.z]
                    point_cloud.points.append(point)
                    point_cloud.colors.append(color)
        return point_cloud

    def save_point_cloud(self, filename):
        # 创建点云
        point_cloud = o3d.geometry.PointCloud()
        # 获取图片样式大小
        color_image, depth_img = self.get_frames()
        # 获取深度帧 
        depth_frame = self.kinect.get_last_depth_frame()
        # 获取相机空间点
        camera_space_points = self.get_camera_space_points(depth_frame)
        # 获取颜色空间点
        color_space_points = self.get_color_space_points(depth_frame)
        # 为每个深度点分配颜色
        with open(filename+'camera_space_points.txt', 'w') as f1, open(filename+'color_space_points.txt', 'w') as f2:
            for i, point in enumerate(camera_space_points):
                x, y, z = float(point.x), float(point.y), float(point.z)
                if not (np.isinf(x) or np.isinf(y) or np.isinf(z)):  # 如果点的坐标不是inf或-inf
                    color_space_point = color_space_points[i]
                    x_c, y_c = int(color_space_point.x), int(color_space_point.y)
                    if 0 <= x_c < color_image.shape[1] and 0 <= y_c < color_image.shape[0]:
                        color = color_image[y_c, x_c]/255.0
                        point = [point.x, point.y, point.z]
                        point_cloud.points.append(point)
                        point_cloud.colors.append(color)
                        # 将点的坐标写入camera_space_points.txt
                        f1.write(f"{point[0]} {point[1]} {point[2]}\n")
                        # 将颜色空间点的坐标写入color_space_points.txt
                        f2.write(f"{float(color_space_point.x)} {float(color_space_point.y)}\n")
        # 保存颜色帧图片
        cv2.imwrite(filename + 'result.png', color_image)
        # 保存点云
        o3d.io.write_point_cloud(filename+'result.pcd', point_cloud)

    def close(self):
        # 释放资源
        self.kinect.close()

    # 在处理图像之后，根据像素图象的point找到相机空间中对应的三维点
    def search_3dImgIndex(self,pred):
        # 假设这里图像是只有目标区域的0-1二值化图形
        # 那么我们就需要找到这个图形的最大轮廓也就是目标区域的边界
        # 这里我们使用cv2.findContours函数来找到这个边界
        # 假设只有一个目标区域，不然就需要找到最大的那个(暂时没有开发)
        # color_img = cv2.cvtColor(color_result_image, cv2.COLOR_BGR2GRAY)
        # 归一化处理，这里应该将结果哪里的0-1值转换为0-255，这里再除以255，显得有点多余
        # color_img = color_result_image/255
        # color_img = color_img.astype(np.uint8)
        # binary_img = np.uint8(pred > 0.5) * 255
        edgePoints = []        
        binary_img = np.uint8(pred)

        # 最大轮廓边缘点代码
        # contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # # 找到轮廓后，保存下轮廓上每个像素点的位置信息
        # # 获取最大连通域的边缘点
        # maxContourIdx = 0
        # if contours:
        #     maxContourArea = cv2.contourArea(contours[0])
        #     for i in range(1, len(contours)):
        #         contourArea = cv2.contourArea(contours[i])
        #         if contourArea > maxContourArea:
        #             maxContourArea = contourArea
        #             maxContourIdx = i
        #     # 保存最大连通域的边缘点
        #     edgePoints = contours[maxContourIdx]
        # print("the wounds edges(pixel) size is: ", len(edgePoints))

        # 寻找连通域  保存最大连通域内所有点，只有边缘点太少了，这里保存所有点
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
        # 获取标签为0的所有像素的坐标，也就是黑色，白色标签为1
        y_coords, x_coords = np.where(labels == 0)
        # 将x和y坐标存储在两个数组中
        edgePoints = np.column_stack((x_coords, y_coords))
        print("the wounds area(pixel) size is: ", len(edgePoints))

        # 特别标明，这里的坐标是图像坐标，不是相机空间坐标，且数量为0时，返回空列表
        if len(edgePoints)==0:
            return []
        # 获取深度帧 
        depth_frame = self.kinect.get_last_depth_frame()
        #获取颜色帧待搜寻空间点的坐标 
        color_space_points = self.get_color_space_points(depth_frame)
        # 获取深度帧映射的相机空间映射点
        camera_space_points = self.get_camera_space_points(depth_frame)
        camera_space_points = np.array(camera_space_points)
        # 待搜寻索引
        edgePointIndex2d = SortedSet()

        cloud = []
        for i in range(len(color_space_points)):
            cloud.append([color_space_points[i].x, color_space_points[i].y, 0])
        cloud = np.array(cloud, dtype=np.float32)
        cloud = pcl.PointCloud.PointXYZ.from_array(cloud)

        kdtree = pcl.kdtree.KdTreeFLANN.PointXYZ()
        kdtree.setInputCloud(cloud)
        # k最近邻搜索
        k = 4
        pointIdxNKNSearch = pclpy.pcl.vectors.Int([0] * k)
        pointNKNSquaredDistance = pclpy.pcl.vectors.Float([0] * k)
        for i in range(len(edgePoints)):
            searchPoint = pcl.point_types.PointXYZ()
            # 传进来的是一个roi区域大小的图像，这里原始投影是1920x1080，
            # 但这里的坐标是roi区域的坐标，所以需要转换下，或者在传进来的时候就需要更换
            # 直接传进来一个1920x1080大小图像，免得roi不同，还需要修改这里的代码--2024.6.13
            searchPoint.x = edgePoints[i][0]
            searchPoint.y = edgePoints[i][1]
            searchPoint.z = 0
            # 对于每个点进行搜寻
            if kdtree.nearestKSearch(searchPoint, k, pointIdxNKNSearch, pointNKNSquaredDistance) > 0:
                # 索引和距离  默认以距离排序，从小到大，所以取距离最近的一个点，也就是其索引
                most_near = pointIdxNKNSearch[0]
                # 这里保存了轮廓像素点找到最近点的2d索引，之后应该是在三维空间中得到其坐标，根据索引一一对应的关系
                edgePointIndex2d.add(most_near)
        print("the wounds finally area(pixel) size is: ", len(edgePointIndex2d))
        # 遍历完边缘点索引
        wound_point_3d = []
        # 测试代码
        with open('data\\points\\test_roi_3d\\camera_space_points.txt', 'w') as f1:
            for i in edgePointIndex2d:
                point = camera_space_points[i]
                wound_point_3d.append([point[0], point[1], point[2]])
                # 将点的坐标写入camera_space_points.txt
                f1.write(f"{point[0]} {point[1]} {point[2]}\n")

        # for i in edgePointIndex2d:
        #     point = camera_space_points[i]
        #     wound_point_3d.append([point[0], point[1], point[2]])
            # 将点的坐标写入camera_space_points.txt
            # f1.write(f"{point[0]} {point[1]} {point[2]}\n")
        # 保存颜色帧图片
        # cv2.imwrite(filename + 'result.png', color_result_image)
        return wound_point_3d

    def get_predict_wound_edge(self, pred):
            binary_img = np.uint8(pred)
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour_area = cv2.contourArea(contours[0])
                max_contour_idx = 0
                for i in range(1, len(contours)):
                    contour_area = cv2.contourArea(contours[i])
                    if contour_area > max_contour_area:
                        max_contour_area = contour_area
                        max_contour_idx = i
                # 创建白色背景图像
                background = np.ones_like(binary_img) * 255
                # 在背景图像上绘制轮廓
                cv2.drawContours(background, contours, max_contour_idx, (0, 0, 255), thickness=cv2.FILLED)
                only_wound_shape = background.copy()
                # 获取轮廓的中心点
                contour = contours[max_contour_idx]
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                # 分段处理轮廓
                segment_points = []
                for i in range(len(contour)):
                    p1 = contour[i][0]
                    p2 = contour[(i + 1) % len(contour)][0]
                    vec = p2 - p1
                    length = np.linalg.norm(vec)
                    num_segments = int(length / 7)
                    if num_segments > 0:
                        segment_vec = vec / num_segments
                        for j in range(num_segments):
                            suture_point = p1 + (j + 0.5) * segment_vec
                            # 计算当前点的法向量（顺时针旋转90度和逆时针旋转90度）
                            normal_vec1 = np.array([-vec[1], vec[0]], dtype=np.float32)
                            normal_vec2 = np.array([vec[1], -vec[0]], dtype=np.float32)
                            # 归一化法向量
                            normal_vec1 /= np.linalg.norm(normal_vec1)
                            normal_vec2 /= np.linalg.norm(normal_vec2)
                            # 在当前点沿着法向量方向生成两个缝合点，并加上偏移量
                            suture_point1 = suture_point + 10 * normal_vec1
                            suture_point2 = suture_point + 10 * normal_vec2
                            segment_points.append(suture_point1.astype(int))
                            segment_points.append(suture_point2.astype(int))
                    # 在背景图像上绘制分段点，黑色
                for point in segment_points:
                    if 0 <= point[1] < background.shape[0] and 0 <= point[0] < background.shape[1]:
                        background[point[1], point[0]] = 0
                return only_wound_shape,background
            else:
                return None
            
    def getRm65RunPoints(data):
        x = data[:, 0]
        y = data[:, 1]
        # 对数据进行排序，确保按照 x 的顺序排列
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        y = y[sorted_indices]
        # 样条插值
        tck, u = splprep([x, y], s=1)
        x_new, y_new = splev(np.linspace(0, 1, num=500), tck)
        # 分段数
        num_segments = 10
        segment_length = len(x_new) // num_segments
        # 计算每段的中心点
        center_points = []
        for i in range(num_segments):
            mid_index = i * segment_length + segment_length // 2
            center_points.append([x_new[mid_index], y_new[mid_index]])
        center_points = np.array(center_points)
        # 生成从点
        offset = 0.01  # 从点偏移量
        slave_points1 = center_points + np.array([0, offset])
        slave_points2 = center_points - np.array([0, offset])
        # 将z坐标设为固定值，例如z=0
        # z_fixed = 60.0
        # center_points_3d = np.hstack((center_points, np.full((center_points.shape[0], 1), z_fixed)))
        # # 计算切线方向并生成旋转矩阵
        # rotations = []
        # for i in range(1, len(center_points_3d) - 1):
        #     p1 = center_points_3d[i - 1]
        #     p2 = center_points_3d[i + 1]
        #     direction = p2 - p1
        #     direction /= np.linalg.norm(direction)
        #     # 默认 z 轴为固定方向，可以根据实际需要调整
        #     z_axis = np.array([0, 0, 1])
        #     # 创建旋转矩阵，使x轴与方向对齐，y轴与z_axis正交，z轴为z_axis
        #     x_axis = direction
        #     y_axis = np.cross(z_axis, x_axis)
        #     y_axis /= np.linalg.norm(y_axis)
        #     z_axis = np.cross(x_axis, y_axis)
        #     rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
        #     rotations.append(R.from_matrix(rotation_matrix).as_quat())

        # # 打印六维位姿数据
        # for i, point in enumerate(center_points_3d[1:-1]):
        #     print("Position:", point)
        #     print("Orientation (quaternion):", rotations[i])
        #     print()
        # 使用matplotlib绘制图像
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o', label='origin data')
        ax.plot(x_new, y_new, '-', label='b line')
        ax.plot(center_points[:, 0], center_points[:, 1], 'x', label='center')
        ax.plot(slave_points1[:, 0], slave_points1[:, 1], '>', label='s1')
        ax.plot(slave_points2[:, 0], slave_points2[:, 1], '<', label='s2')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('picture')
        ax.axis('equal')  # 保持 x 和 y 轴比例相同
        # 将matplotlib绘图转化为numpy数组
        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        # 转换颜色格式从RGB到BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 关闭matplotlib figure以释放资源
        plt.close(fig)

        return image

def get_rm65_points():
    # img_data_2d
    data = np.loadtxt('data\\points\\test_roi_3d\\camera_space_points.txt', usecols=(0, 1))
    x = data[:, 0]
    y = data[:, 1]

    # 对数据进行排序，确保按照 x 的顺序排列
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    # 样条插值
    tck, u = splprep([x, y], s=1)
    x_new, y_new = splev(np.linspace(0, 1, num=500), tck)
    # 分段数
    num_segments = 10
    segment_length = len(x_new) // num_segments

    # 计算每段的中心点
    center_points = []
    for i in range(num_segments):
        mid_index = i * segment_length + segment_length // 2
        center_points.append([x_new[mid_index], y_new[mid_index]])

    center_points = np.array(center_points)

    # 生成从点
    offset = 0.01  # 从点偏移量
    slave_points1 = center_points + np.array([0, offset])
    slave_points2 = center_points - np.array([0, offset])

    # 将z坐标设为固定值，例如z=0
    z_fixed = 60.0
    center_points_3d = np.hstack((center_points, np.full((center_points.shape[0], 1), z_fixed)))

    # 计算切线方向并生成旋转矩阵
    rotations = []
    for i in range(1, len(center_points_3d) - 1):
        p1 = center_points_3d[i - 1]
        p2 = center_points_3d[i + 1]
        direction = p2 - p1
        direction /= np.linalg.norm(direction)
        # 默认 z 轴为固定方向，可以根据实际需要调整
        z_axis = np.array([0, 0, 1])
        # 创建旋转矩阵，使x轴与方向对齐，y轴与z_axis正交，z轴为z_axis
        x_axis = direction
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
        rotations.append(R.from_matrix(rotation_matrix).as_quat())

    # 打印六维位姿数据
    for i, point in enumerate(center_points_3d[1:-1]):
        print("Position:", point)
        print("Orientation (quaternion):", rotations[i])
        print()


    # 可视化
    plt.plot(x, y, 'o', label='origin data')
    plt.plot(x_new, y_new, '-', label='b line')
    plt.plot(center_points[:, 0], center_points[:, 1], 'x', label='center')
    plt.plot(slave_points1[:, 0], slave_points1[:, 1], '>', label='s1')
    plt.plot(slave_points2[:, 0], slave_points2[:, 1], '<', label='s2')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('picture')
    plt.axis('equal')  # 保持 x 和 y 轴比例相同
    plt.show()


def plotPy_CLoss(file_path):
    data1 = np.loadtxt(file_path + 'camera_space_points.txt')
    # 创建三维散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], label='Data 1')
    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('wound shape data')
    ax.legend()
    # 设置三个轴的范围相同
    max_range = np.array([data1[:, 0].max() - data1[:, 0].min(), 
                        data1[:, 1].max() - data1[:, 1].min(), 
                        data1[:, 2].max() - data1[:, 2].min()]).max() / 2.0
    mid_x = (data1[:, 0].max() + data1[:, 0].min()) * 0.5
    mid_y = (data1[:, 1].max() + data1[:, 1].min()) * 0.5
    mid_z = (data1[:, 2].max() + data1[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 显示图形
    plt.show()

                

if __name__ == "__main__":

    # plotPy_CLoss('data\\points\\test_roi_3d\\')
    get_rm65_points()
    exit(0)
    # 使用示例
    processor = KinectCapture()
    # depth_frame = processor.kinect.get_last_depth_frame()
    # result = processor.get_color_space_points(depth_frame)
    # print("result:", result)
    # 保存点云测试代码
    # color_image, depth_image = processor.get_frames()
    # if color_image is not None and depth_image is not None:
    #     start_time = time.time()
    #     point_cloud = processor.get_point_cloud(color_image)
    #     processor.save_point_cloud('D:\\Program Files\\company\\Jinjia\\Projects\\test_data\\112\\')
    #     end_time = time.time()
    #     execution_time = end_time - start_time
    #     print(f"Execution time: {execution_time} seconds")
