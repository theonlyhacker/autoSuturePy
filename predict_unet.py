import numpy as np
import torch
import torch.nn as nn
import os
import time
import csv
import cv2
from model.unet_model import UNet


# 全局变量，存储kinect相机每次闭合之后的图像，也就是机械臂提到最高or拉紧状态
# 初始为伤口原始状态图像
global kinect_img
global wound_length

class predictImg():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = UNet(n_classes=1,n_channels=1)
        self.net.to(device=self.device)
        self.net.load_state_dict(torch.load('pth\\roi\\unet.pth', map_location=self.device))
        self.net.eval()

    # 图像预测
    def predict_img(self,img):
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print("灰度图shape img shape: ",img.shape)
        # 转为batch为1，通道为1，大小为512*512的数组?
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=self.device, dtype=torch.float32)
        # 预测
        pred = self.net(img_tensor)
        # print(pred.shape)
        # 提取结果
        sig = nn.Sigmoid()
        pred = sig(pred)
        pred = np.array(pred.data.cpu()[0])[0]
        # print("-----result----", pred.shape)
        # 将这部分像素从0-1转化为0-255
        # cv2.imwrite("result_python_148.png",pred)
        pred = pred * 255
        return pred

    # 计算目标区域对角线长度和像素点个数 
    def caculate_img(self,predicted_image):
        """
        Calculate the diagonal length and pixel count of the largest connected component in the predicted image.

        Parameters:
        predicted_image (numpy.ndarray): The predicted image.

        Returns:
        diagonal_length (float): The diagonal length of the bounding box of the largest connected component.
        pixel_count (int): The number of pixels in the largest connected component.
        """
        # 二值化图像
        _, binary_image = cv2.threshold(predicted_image, 0.5, 1, cv2.THRESH_BINARY)
        # 高斯模糊
        blurred_image = cv2.GaussianBlur(binary_image, (3,3), 0)
        # 连通组件分析
        _, labels, stats, _ = cv2.connectedComponentsWithStats(blurred_image.astype(np.uint8))

        # 找到面积最大的连通域
        max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        max_component = (labels == max_label).astype(np.uint8)

        # 获取最大连通域的像素点个数
        pixel_count = np.sum(max_component)   

        # 创建新图像，将最大连通域设置为白色，其他部分设置为黑色
        result_image = np.ones_like(predicted_image) * 255
        result_image[max_component > 0] = 0

        # 获取面积最大的连通域的坐标
        nonzero_coordinates = np.column_stack(np.where(max_component > 0))

        # 获取最大连通域的边界框坐标
        y_min, x_min, height ,width= cv2.boundingRect(nonzero_coordinates)
        # print("x_min ",x_min,"y_min ",y_min,"width ",width,"height: ",height)
        # 计算对角线长度
        diagonal_length = np.sqrt(width**2 + height**2)
        # 在结果图像上绘制最大连通域的边界框（绿色）
        # cv2.rectangle(result_image, (x_min, y_min), (x_min + width, y_min + height), (0,0,255), 2)

        # cv2.imshow("result_image",result_image)
        # cv2.waitKey(0)

        print(f"拟合边框对角线长度：{diagonal_length},像素点个数：{pixel_count}")
        return diagonal_length, pixel_count


if __name__ == "__main__":
    predictImg = predictImg()
    # 单张图测试
    start_time = time.time()
    # predict_image = cv2.imread("data\\data\\April\\4-18\\1st\\collect_1\\152_roiColor.jpg")
    predict_image = cv2.imread("data\\data\\April\\4-28\\top\\origin_1.png")
    # x,y,w,h = 1266,674,165,35
    x,y,w,h = 1255,677,180,35
    roi = predict_image[y:y+h, x:x+w].copy()
    result = predictImg.predict_img(roi)
    cv2.imwrite("113.png",result)
    end_time = time.time()
    print("time: ",end_time-start_time)
    # init_wound_length ,init_pixel_count= predictImg.caculate_img(result)
    # print("init_wound_length: ",init_wound_length,"init_pixel_count: ",init_pixel_count)
