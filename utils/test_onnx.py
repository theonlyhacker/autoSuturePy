import numpy as np
import torch
import torch.nn as nn
import os
import time
import csv
import cv2
from model.unet_model import UNet
from Qt.collect_img import KinectCapture
import onnx
import onnxruntime


# 导出onnx模型，抽离方法（未测试）
def export_onnx():
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_classes=1,n_channels=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('pth\\normal_modelunet.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取图像，实际上只用到了其大小数据，对其实际内容并没有引用 
    img = cv2.imread('data\\normal_distance\\data\\train\\image\\origin_64_original.jpg')
    # get roi
    x,y,w,h= 1020,500,300,130
    img = img[y:y+h, x:x+w]
    # 转为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 转为batch为1，通道为1，大小为512*512的数组?
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
    print("the img of roi",img.shape)
    # 转为tensor
    img_tensor = torch.from_numpy(img)
    # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    
    # add the export onnx model by liuchao
    input_name = 'input'
    output_name = 'output'
    torch.onnx.export(net,img_tensor,'data\\onnx\\unet_lc.onnx',
                         opset_version=12,
                         input_names=[input_name],
                         output_names=[output_name])
    print("export onnx successfully")


# 直接从权重文件进行每张图片的预测，中间有一部分是用来生成onnx模型的代码
def predict_img(img):
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_classes=1,n_channels=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('pth\\normal_modelunet.pth', map_location=device))
    # 测试模式
    net.eval()
    # get roi
    x,y,w,h= 1020,500,300,130
    img = img[y:y+h, x:x+w]
    # 转为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 转为batch为1，通道为1，大小为512*512的数组?
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
    print("the img of roi",img.shape)
    # 转为tensor
    img_tensor = torch.from_numpy(img)
    # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    
    # add the export onnx model by liuchao
#     input_name = 'input'
#     output_name = 'output'
#     torch.onnx.export(net,img_tensor,'unet_lc.onnx',
#                          opset_version=12,
#                          input_names=[input_name],
#                          output_names=[output_name])
#     exit(0)
    # 预测
    pred = net(img_tensor)
    print("\n---------------pred shape--------------", pred.shape)
    # 提取结果
    sig = nn.Sigmoid()
    pred = sig(pred)
    pred = np.array(pred.data.cpu()[0])[0]
    # print("-----result----", pred.shape)
    # 将这部分像素从0-1转化为0-255
    pred = pred * 255
    return pred

# 计算每张图片最大连通域的长度，像素点信息
def caculate_img(predicted_image):
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
    print("x_min ",x_min,"y_min ",y_min,"width ",width,"height: ",height)
    # 计算对角线长度
    diagonal_length = np.sqrt(width**2 + height**2)
    # 在结果图像上绘制最大连通域的边界框（绿色）
    # cv2.rectangle(result_image, (x_min, y_min), (x_min + width, y_min + height), (0,0,255), 2)

    # cv2.imshow("result_image",result_image)
    # cv2.waitKey(0)

    print(f"拟合边框对角线长度：{diagonal_length},像素点个数：{pixel_count}")
    return diagonal_length, pixel_count

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hardsigmoid(x):
    return np.clip((x + 3) / 6, 0, 1)

# 通过生成的onnx模型来预测每次得到最大连通域
def load_onnx():
    # Load the ONNX model
    onnx_model = onnx.load("data\\onnx\\unet_lc.onnx")
    # Verify the ONNX model
    onnx.checker.check_model(onnx_model)

    # Create an ONNX Runtime session
    ort_session = onnxruntime.InferenceSession("data\\onnx\\unet_lc.onnx")
    # 读取输入图片（示例：使用OpenCV读取）
    input_image = cv2.imread('data\\normal_distance\\data\\train\\image\\origin_64_original.jpg')
    # 根据模型的输入要求，截取相应的ROI并转换格式
    x, y, w, h = 1020, 500, 300, 130
    roi = input_image[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     roi_resized = cv2.resize(roi_gray, (512, 512))
    # 将输入转为 ONNX Runtime 期望的格式（NCHW，float32）
    input_data = np.expand_dims(np.expand_dims(roi_gray, axis=0), axis=0).astype(np.float32)
    # print("the input data of input_data is ",input_data)
    # 执行推理
    outputs = ort_session.run(None, {'input': input_data})
    # 获取输出结果
    output_result = outputs[0]
    print("before squeeze the shape of result is ",output_result.shape)
    # 将输出结果归一化到 [0, 1] 范围
    output_result = np.squeeze(output_result)
    print("after squeeze of result is ",output_result)
    # 使用 Sigmoid 函数进行归一化
    output_result = sigmoid(output_result)
    # print("--------output_result-----------",output_result.shape)
    # cv2.imwrite("result_onnx_sigmoid.png",output_result)
    pred = output_result*255
    cv2.imshow("pred",pred)
    cv2.waitKey(0)
    return pred

# 用来计算通过传统方式得到最大连通域的像素点个数
def caculate_standred_img(filepath):
    origin_img = cv2.imread(filepath)
    output_img = origin_img.copy()
    x, y, w, h = 1068, 547, 139, 30
    roi = origin_img[y:y+h, x:x+w].copy()
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 进行高斯滤波
    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    # 进行边缘检测
    edges = cv2.Canny(blurred, 50, 65, apertureSize=3)

    # 对边缘进行膨胀和腐蚀处理，迭代3次
    kernel = np.ones((2,2),np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    # edges = cv2.erode(edges, kernel, iterations=1)

    # 连通组件分析
    _, labels, stats, _ = cv2.connectedComponentsWithStats(edges.astype(np.uint8))

    # 找到面积最大的连通域
    max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    max_component = (labels == max_label).astype(np.uint8)

    # 提取最大连通域的边缘点
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 保存最大轮廓区域内的像素坐标
    maxContourPoints = []
    if len(contours) > 0:
        maxContourPoints = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]
    # 创建一个与输入图像大小相同的二值图像，用于将最大轮廓内的像素点标记为白色
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    if len(maxContourPoints) > 0:
        cv2.drawContours(mask, [maxContourPoints], -1, 255, cv2.FILLED)
    pixel_count = 0
     # 提取最大轮廓内的像素点
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 255:
                origin_x = j + x
                origin_y = i + y
                pt = (origin_x, origin_y)  # 修正坐标顺序
                output_img = cv2.circle(output_img, pt, 1, (255, 255, 255), -1)  # 使用圆形标记像素点为白色
                pixel_count = pixel_count + 1
    cv2.imshow("output_img",output_img)
    cv2.waitKey(0)
    # 获取最大连通域的像素点个数
    print("------lc of pixel------", pixel_count)
    return

if __name__ == "__main__":
    # caculate_standred_img("data\\normal_distance\\data\\train\\image\\origin_64_original.jpg")
    # exit()
    result = load_onnx()
    # exit(0)
    line_length, pixel_count = caculate_img(result)
    print("line_length: ",line_length,"pixel_count: ",pixel_count)
    exit(0)
#     predict_image = cv2.imread("data\\normal_distance\\data\\train\\image\\origin_64_original.jpg")
#     result = predict_img(predict_image)
#     exit(0)
    # 链接相机
    kinect_capture = KinectCapture()
    # 初始化 CSV 文件
    csv_file_path = 'data\\csv\\wound_data_v4.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['Frame', 'Diagonal Length', 'Pixel Count'])

    frame_count = 0  # 记录帧数
    init_cal_count = 0  # 记录初始化次数
    init_wound_length = 0  # 记录初始伤口长度
    init_pixel_count = 0  # 记录初始像素点个数
    
    while True:
        # 获取当前相机图片
        color_frame ,depth_frame= kinect_capture.get_frames()
        if color_frame is not None and depth_frame is not None:
            # 用作第一次or初始位置时计算其长度
            if init_cal_count == 0:
                init_predict_img = predict_img(color_frame)
                init_wound_length ,init_pixel_count= caculate_img(init_predict_img)
                print("------------init length---------------")
                init_cal_count = +1
            # 正常判断条件进行
            predict_color_img = predict_img(color_frame)
            # 计算当前图片拟合长度以及像素点个数信息
            line_length, pixel_count = caculate_img(predict_color_img)
            # print("----------caculate over----------\n",line_length)

            # 保存数据到 CSV 文件
            frame_count += 1
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_count, line_length, pixel_count])
            # 判断条件
            if (init_wound_length - line_length)>10 and (init_pixel_count - pixel_count)>200:
                # 添加其他判断条件or返回缝合成功状态码
                init_wound_length = line_length
                init_pixel_count = pixel_count
                code = True
                print("should stop RM65")
        # 每秒钟取5张进行判断计算
        time.sleep(0.2)

    # print()