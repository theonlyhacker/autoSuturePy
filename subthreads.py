import sys
from PyQt5.QtCore import QThread
import cv2
from predict_unet import predictImg
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer,Qt,QThread,pyqtSignal,pyqtSlot
import ctypes
import serial
import time
import os
import binascii
import struct
from statusModel.cnn import CustomCNN,predict_status
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# roi区域--统一在这里进行更改--也可以用信号和槽的方式进行传递？？？
x,y,w,h = 1190,730,180,50
# Lookup table for CRC calculation
aucCRCHi = [
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40
]

aucCRCLo = [
    0x00, 0xC0, 0xC1, 0x01, 0xC3, 0x03, 0x02, 0xC2, 0xC6, 0x06, 0x07, 0xC7,
    0x05, 0xC5, 0xC4, 0x04, 0xCC, 0x0C, 0x0D, 0xCD, 0x0F, 0xCF, 0xCE, 0x0E,
    0x0A, 0xCA, 0xCB, 0x0B, 0xC9, 0x09, 0x08, 0xC8, 0xD8, 0x18, 0x19, 0xD9,
    0x1B, 0xDB, 0xDA, 0x1A, 0x1E, 0xDE, 0xDF, 0x1F, 0xDD, 0x1D, 0x1C, 0xDC,
    0x14, 0xD4, 0xD5, 0x15, 0xD7, 0x17, 0x16, 0xD6, 0xD2, 0x12, 0x13, 0xD3,
    0x11, 0xD1, 0xD0, 0x10, 0xF0, 0x30, 0x31, 0xF1, 0x33, 0xF3, 0xF2, 0x32,
    0x36, 0xF6, 0xF7, 0x37, 0xF5, 0x35, 0x34, 0xF4, 0x3C, 0xFC, 0xFD, 0x3D,
    0xFF, 0x3F, 0x3E, 0xFE, 0xFA, 0x3A, 0x3B, 0xFB, 0x39, 0xF9, 0xF8, 0x38,
    0x28, 0xE8, 0xE9, 0x29, 0xEB, 0x2B, 0x2A, 0xEA, 0xEE, 0x2E, 0x2F, 0xEF,
    0x2D, 0xED, 0xEC, 0x2C, 0xE4, 0x24, 0x25, 0xE5, 0x27, 0xE7, 0xE6, 0x26,
    0x22, 0xE2, 0xE3, 0x23, 0xE1, 0x21, 0x20, 0xE0, 0xA0, 0x60, 0x61, 0xA1,
    0x63, 0xA3, 0xA2, 0x62, 0x66, 0xA6, 0xA7, 0x67, 0xA5, 0x65, 0x64, 0xA4,
    0x6C, 0xAC, 0xAD, 0x6D, 0xAF, 0x6F, 0x6E, 0xAE, 0xAA, 0x6A, 0x6B, 0xAB,
    0x69, 0xA9, 0xA8, 0x68, 0x78, 0xB8, 0xB9, 0x79, 0xBB, 0x7B, 0x7A, 0xBA,
    0xBE, 0x7E, 0x7F, 0xBF, 0x7D, 0xBD, 0xBC, 0x7C, 0xB4, 0x74, 0x75, 0xB5,
    0x77, 0xB7, 0xB6, 0x76, 0x72, 0xB2, 0xB3, 0x73, 0xB1, 0x71, 0x70, 0xB0,
    0x50, 0x90, 0x91, 0x51, 0x93, 0x53, 0x52, 0x92, 0x96, 0x56, 0x57, 0x97,
    0x55, 0x95, 0x94, 0x54, 0x9C, 0x5C, 0x5D, 0x9D, 0x5F, 0x9F, 0x9E, 0x5E,
    0x5A, 0x9A, 0x9B, 0x5B, 0x99, 0x59, 0x58, 0x98, 0x88, 0x48, 0x49, 0x89,
    0x4B, 0x8B, 0x8A, 0x4A, 0x4E, 0x8E, 0x8F, 0x4F, 0x8D, 0x4D, 0x4C, 0x8C,
    0x44, 0x84, 0x85, 0x45, 0x87, 0x47, 0x46, 0x86, 0x82, 0x42, 0x43, 0x83,
    0x41, 0x81, 0x80, 0x40
]
# Helper function to compute CRC16
def _bMBCRC16(pucFrame, usLen):
    ucCRCHi = 0xFF
    ucCRCLo = 0xFF
    while usLen > 0:
        iIndex = ucCRCLo ^ pucFrame[0]
        pucFrame = pucFrame[1:]
        ucCRCLo = ucCRCHi ^ aucCRCHi[iIndex]
        ucCRCHi = aucCRCLo[iIndex]
        usLen -= 1
    return (ucCRCHi << 8) | ucCRCLo

# Helper function to check the frame validity
def WRIST_FrmCheck(RcvDataBuff, RcvLen):
    cur_off = 0
    while cur_off + 7 <= RcvLen:
        if RcvDataBuff[cur_off] != 0x01 or RcvDataBuff[cur_off + 1] != 0x03:
            cur_off += 1
            continue
        crc = (RcvDataBuff[cur_off + 6] << 8) | RcvDataBuff[cur_off + 5]
        if crc != _bMBCRC16(RcvDataBuff[cur_off:cur_off + 5], 5):
            cur_off += 1
            continue
        return cur_off
    return -1

# Main function to parse the sensor data, modbus protocol
def WRIST_FrmPrase(pRcvBuf, nRcvLen):
    cur_off = 0
    if pRcvBuf is None or nRcvLen == 0:
        return False, []
    pOutDataArry = []
    while cur_off + 7 <= nRcvLen:
        tmp_off = WRIST_FrmCheck(pRcvBuf[cur_off:], nRcvLen - cur_off)
        if tmp_off < 0:
            break
        cur_off += tmp_off
        tmp = struct.unpack('>h', bytes(pRcvBuf[cur_off + 3:cur_off + 5]))[0]
        pOutDataArry.append(tmp / 10.0)
        cur_off += 7
    return True, pOutDataArry

# 缝合装置运动--仅在自动缝合中使用该函数？--感觉有问题，自动缝合出发的电机端口是这个吗？
def motorRun():
    print("success into 电机")
    # 创建串口对象
    ser = serial.Serial()
    # 配置串口参数
    ser.port = 'COM3'
    ser.baudrate = 115200
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE
    ser.timeout = 5  # 超时时间设置为5秒
    # 打开串口
    try:
        ser.open()
    except serial.SerialException as e:
        print(f"Failed to open serial port: {e}")
        exit(1)
    # 发送数据
    send_data = b's'
    ser.write(send_data)
    # 等待数据发送完成
    time.sleep(1)  # 等待1秒，确保数据发送完成
    # 接收数据
    read_data = bytearray()
    received_complete_data = False
    timeout = 10  # 设置一个合适的超时时间，这里设为10秒
    start_time = time.time()
    while not received_complete_data and (time.time() - start_time) < timeout:
        if ser.in_waiting > 0:
            read_data.extend(ser.read(ser.in_waiting))
            print(read_data)
            if b"motor_stop_signal" in read_data:
                received_complete_data = True
    if received_complete_data:
        # 处理接收到的数据
        print("Received data:", read_data)
        return "motor_run_successful"
    else:
        print("Timeout occurred while waiting for complete data.")
    # 关闭串口
    ser.close()
    return "motor_run_defualt"

#   调用结构体 POSE
class DevMsg(ctypes.Structure):
    _fields_ = [("px", ctypes.c_float),
                ("py", ctypes.c_float),
                ("pz", ctypes.c_float),
                ("rx", ctypes.c_float),
                ("ry", ctypes.c_float),
                ("rz", ctypes.c_float)]
#   调用结构体 POSE 
class DevMsge(ctypes.Structure):
    _fields_ = [("frame_name", ctypes.c_char * 10),
                ("pose", DevMsg),
                ("payload", ctypes.c_float),
                ("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("z", ctypes.c_float)]

# 更新显示区域子线程-包括缝合到顶点规划显示
class show_roi_thread(QThread):
    show_status_signal = pyqtSignal(int)

    def  __init__(self,main_thread):
        super().__init__()
        self.main_thread = main_thread
        self.kinect = main_thread.kinect
        self.rm65 = main_thread.rm65
        self.show_roi_status = True
        self.show_status = 0 #默认从初始位姿运动之后是向下运动

        self.model = CustomCNN((3, 40, 180))
        self.model.load_state_dict(torch.load('pth\\status\\status_cnn.pth'))# 加载预训练模型
        self.model.eval()

        self.predict_roi_img = predictImg()

        self.show_status_signal.connect(self.change_show_status)

    def run(self):
        # 这个思路就是保持该线程一直在循环中，利用信号来控制是抬升阶段还是其他阶段的渲染，这样就可以实现不同阶段的渲染
        self.count =0
        transform = transforms.Compose([transforms.ToTensor() ])# 转换为张量
        consecutive_ones_count = 0
        final_img = None  # 用于叠加三张图像的变量
        origin_img = None
        while self.show_roi_status:
            color,depth = self.kinect.get_frames()
            # 找到的第一张非空图片用来跟新视觉显示区域，其他的用来判断状态
            if color is not None:
                roi_img = color[y:y+h, x:x+w].copy()
                if self.show_status == 0:
                    scale_img = roi_img.copy()
                    height, width, channel = scale_img.shape
                    bytes_per_line = 3 * width# 将图像转换为QImage,三通道的用这个方法
                    q_image = QImage(scale_img.data, width, height,bytes_per_line, QImage.Format_BGR888)
                    self.main_thread.label_img.setPixmap(QPixmap.fromImage(q_image))# 在 QLabel 中显示图像，并使用填充方式
                    self.main_thread.label_img.setScaledContents(True)

                elif self.show_status == 1 and consecutive_ones_count < 3:
                    # print(f"current consecutive_ones_count: {consecutive_ones_count}")
                    # roi_img = cv2.imread("data\\data\\status_train\\4-26-img\\collect_2\\169_roi_1.jpg")
                    pred = self.predict_roi_img.predict_img(roi_img)
                    binary_img = np.uint8(pred)
                    if final_img is None:
                        final_img = binary_img
                    else:
                        final_img = final_img | binary_img  # 按位或运算符
                    # 连续获取三次kinect图像进而进行路径规划--防止单张图像预测效果不好,以或运算，只要预测到的roi区域都渲染
                    consecutive_ones_count +=1
                    if consecutive_ones_count ==2:
                        wound_shape_data,multi_pred_img = self.kinect.get_predict_wound_edge(final_img)# 获取预测的伤口边缘数据
                        height, width = multi_pred_img.shape
                        bytes_per_line = width
                        q_image = QImage(multi_pred_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                        # 在 QLabel 中显示图像，并保持原比例
                        self.main_thread.label_suturePoint.setPixmap(
                            QPixmap.fromImage(
                                q_image.scaled(self.main_thread.label_suturePoint.size(), 
                                            aspectRatioMode=Qt.KeepAspectRatio)))
                        self.show_status = 0
                        consecutive_ones_count = 0
                        multi_pred_img = None
                        final_img = None
                elif self.show_status == 2:
                    # 这个阶段要一边渲染一边判断状态
                    self.count += 1
                    # print(f"此次判断次数为{self.count}")
                    image = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)# 图像预处理转换
                    if self.count % 2 == 0: #每两次预测渲染一次roi页面
                        scale_img = roi_img.copy()
                        height, width, channel = scale_img.shape # 将图像转换为QImage
                        bytes_per_line = 3 * width
                        q_image = QImage(scale_img.data, width, height,bytes_per_line, QImage.Format_BGR888)
                        self.main_thread.label_img.setPixmap(QPixmap.fromImage(q_image))# 在 QLabel 中显示图像，并使用填充方式
                        self.main_thread.label_img.setScaledContents(True)
                    image = Image.fromarray(image)# 将图像转换为 PIL 图像
                    image1 = transform(image) #将图像转成张量
                    # predict_status_result = predict_status(model=self.model,image=image1) # 预测图像
                    # print(f"currrent status predict : {predict_status_result}")
                    # 判断预测结果
                    # if predict_status_result == 1:
                    #     consecutive_ones_count += 1
                    # else:
                    #     consecutive_ones_count = 0
                    # 如果连续三次预测结果为1，则执行停止指令
                    # if consecutive_ones_count >= 2:
                    #     print("预测结果为：",predict_status_result) 
                    #     print("达到阈值，发射停止机械臂运动信号--机械臂抬升停止--由状态模型判断")
                    #     consecutive_ones_count = 0
                    #     self.count = 0
                    #     self.rm65.pDll.Move_Stop_Cmd(self.rm65.nSocket,1)#停止机械臂运动
                    #     self.change_show_status(1)
                
                elif self.show_status == 3:
                    # 暂时用于路径点规划--标定等函数
                    pred = self.predict_roi_img.predict_img(roi_img)
                    binary_img = np.uint8(pred)
                    wound_shape_data,multi_pred_img = self.kinect.get_predict_wound_edge(binary_img)# 获取预测的伤口边缘数据
                    height, width = multi_pred_img.shape
                    # 将得到的pred图像扩充为1920x1080大小的图像，然后保存其roi位置信息到本地，然后进行函数拟合得到分段点
                    # 创建一个白色的图像，大小为1920x1080--由于是二值化图像，因此创建模板是单通道的
                    final_canvas = np.ones((1080, 1920), dtype=np.uint8) * 255
                    final_canvas[y:y+h, x:x+w] = wound_shape_data
                    # final_canvas = cv2.bitwise_not(final_canvas)
                    # cv2.imwrite('data\\points\\test_roi_3d\\final_output.png', final_canvas)# 保存最终图像
                    wound_point_3d=self.kinect.search_3dImgIndex(final_canvas)
                    # 由于目前伤口都在一个平面上，因此投影到一个面上进行拟合（用转换矩阵之前）
                    func_data=[]
                    for i in wound_point_3d:
                        func_data.append([i[0],i[1],0.160])
                    # print(func_data)
                    show_programming_points = self.kinect.getTurePointsRm65(func_data)
                    print(show_programming_points)
                    self.change_show_status(1)

    @pyqtSlot(int)
    def change_show_status(self,status):
        if hasattr(self,'count'):
            print(f"状态判断次数: {self.count}")
        # 将切换渲染状态简化为由槽函数与信号，使得程序逻辑更加简单明了，同时经测试实际运行时长并没有明显的增加？
        print(f"进入路径规划渲染槽函数,status: {status}")
        self.show_status = status
        self.count = 0

    def stop(self):
        print("渲染子线程结束")
        self.show_roi_status = False
    

# 示教模式下，读取压力传感器读数子线程
class PressureThread(QThread):
    pressusre_stop_signal = pyqtSignal()

    def __init__(self,work_thread,filename):
        super().__init__()
        self.work_thread = work_thread
        self.filename = filename
        self.stop_requested = False
        self.pressusre_stop_signal.connect(self.stop)
    
    def run(self):
        # send_data = binascii.unhexlify("01030000000045CA")#多次接受数据
        # send_data = binascii.unhexlify("010300000001840A")#单次接受数据
        # 收集数据模块
        with open(self.filename+'_pressure.txt', 'a', newline='') as txtfile:
            while not self.stop_requested:
                send_data = binascii.unhexlify("010300000001840A")
                self.work_thread.serial.write(send_data)
                time.sleep(0.01)
                # 解析接收到的数据
                read_data = self.work_thread.serial.read_all()
                rcv_buf = list(read_data)
                rcv_len = len(rcv_buf)
                success ,rcvData= WRIST_FrmPrase(rcv_buf, rcv_len)
                if rcvData:  # 检查rcvData是否为空
                    txtfile.write(str(rcvData[0])+'\n')
                    txtfile.flush()  # 强制将缓冲区的数据写入文件
    
    def stop(self):
        self.stop_requested = True

# 收集相机图像数据和压力传感器数据，用于缝合状态是否完好的判断
# 目前的思路是记录每一段向上抬起这一阶段的数据，也只有这一阶段需要来判断--3.19.2024
# 仅收集图像数据--3.27.2024
class collect_Kinect_Pressure(QThread):
    stop_signal = pyqtSignal()
    def __init__(self,main_thread):
        super().__init__()
        self.main_thread = main_thread

        self.stop_requested = False
        self.stop_signal.connect(self.stop)
    
    def run(self):
        # print("收集图像和压力传感器数据")
        main_filename = self.main_thread.record_info + "\\"+"collect_"+str(self.main_thread.record_info_count)
        os.makedirs(os.path.join(main_filename), exist_ok=True)
        record_count = 0
        count_color_null = 0

        # 只记录抬升阶段的压力数据
        # with open(self.main_thread.record_info+'\\pressure_all.txt', 'a', newline='') as txtfile:
        #     while not self.stop_requested:
        #         # 发送指令并接收数据
        #         send_data = binascii.unhexlify("010300000001840A")
        #         self.main_thread.serial.write(send_data)
        #         time.sleep(0.01)
        #         # 解析接收到的数据
        #         read_data = self.main_thread.serial.read_all()
        #         rcv_buf = list(read_data)
        #         rcv_len = len(rcv_buf)
        #         success ,rcvData= WRIST_FrmPrase(rcv_buf, rcv_len)
        #         if rcvData:  # 检查rcvData是否为空
        #             txtfile.write(str(rcvData[0])+'\n')
        #             txtfile.flush()  # 强制将缓冲区的数据写入文件

        while not self.stop_requested:
            # 修改保存文件夹，使其不需要新建文件夹就能保存数据，将一次抬起动作保存到一个文件夹下
            filename = main_filename+"\\"+str(record_count)
            # 先写收集一次的方法，然后在拓展成循环
            color,depth = self.main_thread.kinect.get_frames()
            # 这里存在一个bug，就是color帧有时候获取不稳定，造成数据收集出现问题
            if color is not None:
                # 启动压力传感器线程
                pressure_thread = PressureThread(self.main_thread,filename)
                pressure_thread.start()
                # 保存图像
                colorImg = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
                # roi区域 
                x,y,w,h= 1255,677,180,40
                roi_img = colorImg[y:y+h, x:x+w].copy()
                cv2.imwrite(filename + "_roiColor.jpg",roi_img)
                time.sleep(0.01)
                record_count += 1
                # 结束压力传感器线程
                pressure_thread.stop()
            else:
                # print("cout color is None: ",count_color_null)
                count_color_null += 1

    def stop(self):
        self.stop_requested = True


# 自定义RM65机械臂运动线程类
class WorkThread(QThread):
    stop_signal = pyqtSignal()
    rm65_stop_signal = pyqtSignal()# 用于停止机械臂运动--状态信号的发出

    def __init__(self,main_thread,points):
        super().__init__()
        self.rm65 = main_thread.rm65
        self.kinect = main_thread.kinect
        self.show_kinect_thread = main_thread.show_kinect_thread
        self.main_thread = main_thread

        self.status_thread = None
        self.points = points

        self.stop_signal.connect(self.stop)


    def run(self):
        # 获取机械臂状态--检查当前机械臂坐标系是否修改为suturePin
        tool_name = self.rm65.get_currentToolName()
        if(tool_name != b'suturePin'):
            sys.exit("当前坐标系是:",str(tool_name))
        self.start_time = time.time()
        # 压力传感器先注销--暂时用不到这个进行状态判断  2024.5.8
        # if(not hasattr(self, 'serial')):
        #     self.serial = serial.Serial('COM4', 115200)

        # Movel_Cmd(SOCKHANDLE ArmSocket, POSE pose, byte v, float r, bool block);
        # Movec_Cmd(SOCKHANDLE ArmSocket, POSE pose_via, POSE pose_to, byte v, float r, byte loop, bool block);
        # Movej_P_Cmd(SOCKHANDLE ArmSocket, POSE pose, byte v, float r, bool block);
        data_file_path = "data\\data\\May\\5-21\\movej_P\\"
        point_index = 1
        ret = 0
        with open(data_file_path+"test_update.txt", 'a', newline='') as txtfile:
            for index, point in enumerate(self.points): # 机械臂运动show_wound_path_signal
                self.show_kinect_thread.show_status_signal.emit(1)
                self.rm65.pDll.Movej_P_Cmd(self.rm65.nSocket, point, 50, 0, 1)# 向下运动到待缝合点
                print(f"到达第{point_index}个运动点")
                point_index+=1
                # 开始缝合
                motor_signal = motorRun()
                if motor_signal == "motor_run_successful":
                    print("缝合完成，开始抬升")# 圆弧行运动模块--是标准园还是曲线部分？看具体运动效果
                    temp_point = DevMsg(point.px,point.py,point.pz,point.rx,point.ry,point.rz)
                    temp_point.pz = (21 - index * 2.1) * 0.01 + temp_point.pz
                    temp_point.px -= 0.005
                    # 订书机模式
                    # temp_point.pz = (5) * 0.01 + temp_point.pz
                    # ret = self.rm65.pDll.Movej_P_Cmd(self.rm65.nSocket, temp_point, 50, 0, 1)
                    if index==0:
                        ret = self.rm65.pDll.Movej_P_Cmd(self.rm65.nSocket, temp_point, 50, 0, 1)
                    else:
                        # 第一针之后的缝合，需要启动状态判断子线程，进行状态判断
                        self.show_kinect_thread.show_status_signal.emit(2)
                        # print(f"temp_point: {temp_point.px},py:{temp_point.py},pz: {temp_point.pz},rx:{temp_point.rx},ry:{temp_point.ry},rz:{temp_point.rz}")
                        # print(f"goal_point: {point.px},py:{point.py},pz: {point.pz},rx:{point.rx},ry:{point.ry},rz:{point.rz}")
                        ret = self.rm65.pDll.Movej_P_Cmd(self.rm65.nSocket, temp_point, 50, 0, 1)
                    if ret != 0 :
                        print("第"+str(index)+" 点，该点不可达:" + str(ret))
                        sys.exit()
                print(f"到达第{point_index}个运动点")
                point_index+=1
            self.end_time = time.time()
            txtfile.write(str("50")+"  "+str(self.end_time-self.start_time)+str("")+'\n')
            txtfile.flush()
        print("the spend time is :",self.end_time-self.start_time)
        self.show_kinect_thread.stop()#缝合完成默认停止？
        self.stop()


    def stop(self):
        print("rm65子线程运动完成,结束子线程")
        if  hasattr(self.main_thread, 'timer_pos'):
            print("结束实时位姿展示线程")
            # 定时器的启动和暂停不能跨线程，所以该命令报错
            self.main_thread.timer_pos.stop()
        self.rm65.pDll.Move_Stop_Cmd(self.rm65.nSocket,1)
        sys.exit()
