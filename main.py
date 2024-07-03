from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer,Qt,QThread,pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from Ui_test import Ui_MainWindow
from tools.getKinect import KinectCapture
from tools.rm65 import RobotConnection
from statusModel.cnn import CustomCNN,predict_status
from predict_unet import predictImg
import numpy as np
import serial
import binascii
import time
import sys 
import vtk
import cv2
import ctypes
import os
import struct
import random
import csv
from queue import Queue
from queue import Empty
import torch
import torchvision.transforms as transforms
from PIL import Image
from subthreads import *

   
# kinect 相机参数
rgbWidth,rgbHeight = 1920,1080
depthSize = 512*424
teach_flag = False
# 自定义机械臂运动停止信号
rm65_stop_signal = False
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


# 全局变量，用于保存图像计数 ---仅用作图像采集时的权宜之计
global_count_colorImg = 1
glbal_start_time = 0
class MainWindow(QMainWindow, Ui_MainWindow):
    # 初始化
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.work_thread = None
        self.pressure_thread = None
        self.showWoundStatus = False
        # self.img_roi = [1255,677,180,40] #roi区域,默认为伤口区域
        self.img_roi = [1060,667,185,35] #roi区域,默认为伤口区域
    

    # 获取kinect图像
    def get_colorImg(self):
        if hasattr(self, 'kinect'):
            # 只有在 KinectCapture 实例存在时才调用相应方法
            colorImg,depthImg = self.kinect.get_frames()
            # 检查colorImg是否为None（即图像是否为空）
            if colorImg is not None:
                # 从四通道的RGBA转换为三通道的RGB
                rgb_img = cv2.cvtColor(colorImg, cv2.COLOR_BGRA2BGR)
                x,y,w,h= self.img_roi[0],self.img_roi[1],self.img_roi[2],self.img_roi[3]
                scale_img = rgb_img[y:y+h, x:x+w].copy()
                # scale_img = cv2.resize(roi_img, (480, 270))
                # 将图像转换为QImage
                height, width, channel = scale_img.shape
                bytes_per_line = 3 * width
                q_image = QImage(scale_img.data, width, height,bytes_per_line, QImage.Format_BGR888)
                # 在 QLabel 中显示图像，并使用填充方式
                self.label_img.setPixmap(QPixmap.fromImage(q_image))
                # self.label_img.setAlignment(Qt.AlignCenter)
                self.label_img.setScaledContents(True)
                # 在 QLabel 中显示图像，并保持原比例


    #  高点记录按钮--记录当前伤口信息，伤口2d信息，三维点云信息，压力传感器信息，提起来的高度差
    def on_kinect_pressure(self):
        # 用于到最高点停止收集抬起阶段的图像和传感器数据子线程
        self.img_thread.stop()
        # 输出相关信息
        global glbal_start_time
        end_time = time.time()
        print("the spend time is :",end_time-glbal_start_time)
        self.record_info_count += 1
        print("第 ",self.record_info_count,"  阶段，图像和压力传感器数据采集结束")
        # 保存在最高点时，伤口信息和机械臂位姿信息--方便统计信息
        filename = self.record_info + "\\"+"highSuturePos"
        os.makedirs(os.path.join(filename), exist_ok=True)
        # 记录当前的机械臂位姿信息
        current_pose = self.rm65.get_currentPose()
        with open(filename + '\\sutrue_point_rm65.txt', 'a') as f2:
            f2.write(f"{current_pose.px} {current_pose.py} {current_pose.pz} {current_pose.rx} {current_pose.ry} {current_pose.rz}\n")
        color,depth = self.kinect.get_frames()
        if color is None:
            print("color is None--图像获取失败")
            return
        # 保存图像
        colorImg = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
        # cv2.imwrite(filename+"\\origin_color.jpg",colorImg)
        x,y,w,h= self.img_roi[0],self.img_roi[1],self.img_roi[2],self.img_roi[3]
        roi_img = colorImg[y:y+h, x:x+w].copy()
        cv2.imwrite(filename + f"\\{self.record_info_count}_roi_color.jpg",roi_img)
        print("当前位置所有信息成功记录")


    # 电机测试&&示教模式记录接口  缝合下针按钮
    def on_testSuture(self):
        # 创建串口对象
        ser = serial.Serial('COM3', 115200)
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
        else:
            print("Timeout occurred while waiting for complete data.")
        # 关闭串口
        # ser.close()
        
        # 示教模式下每次缝合都需要将当前位置记录下来，并且需要同步记录此时相机图像，压力传感器等多个数据源（未开发）
        if hasattr(self, 'teach_flag') and hasattr(self, 'rm65'):
            timestamp = int(time.time())
            filename = self.record_info + "\\lowSuturePos"
            os.makedirs(os.path.join(filename), exist_ok=True)
            with open(filename+'\\teach_points_rm65.txt', 'a', newline='') as txtfile:
                result_pose = self.rm65.get_currentPose()
                txtfile.write(str(result_pose.px)+" "+str(result_pose.py)+" "+str(result_pose.pz)
                              +str(result_pose.rx)+" "+str(result_pose.ry)+str(result_pose.rz)+'\n')
                txtfile.flush()  # 强制将缓冲区的数据写入文件
                print("当前位置成功记录")
            # 缝合完成开启图像和压力传感器子线程读数
            # 开启线程，进行图像采集和压力传感器读数
            self.img_thread = collect_Kinect_Pressure(self)
            global glbal_start_time
            glbal_start_time = time.time()
            self.img_thread.start()

    #机械臂初始化（待测试）-参考python-demo3就行，设置工作坐标系啥的 
    # 修改为整个系统的初始化--增加图像显示的初始化以及高处预测轨迹点区域的初始化
    def on_initRM65(self):
        print("开始初始化kinect相机...") 
        if(not hasattr(self, 'kinect')):
            self.kinect = KinectCapture()
            if self.kinect.connect() == None:
                print("kinect相机初始化失败")
            print("kinect相机初始化成功")
        # 机械臂初始化
        if(not hasattr(self, 'rm65')):
            self.rm65 = RobotConnection("192.168.1.19", 8080)
            self.rm65.connect()
        current_tool_name = self.rm65.get_currentToolName()
        print("当前工具坐标系:" + str(current_tool_name))

        float_joint = ctypes.c_float * 6
        joint = float_joint()

        str_buf = ctypes.create_string_buffer("suture".encode('utf-8'))
        self.rm65.pDll.Change_Tool_Frame(self.rm65.nSocket, str_buf, 1)

        current_tool_name = self.rm65.get_currentToolName()
        print("当前工具坐标系(修改之后):" + str(current_tool_name))

        # 启动定时器，用于实时显示机械臂末端位姿
        self.timer_pos = QTimer(self)
        self.timer_pos.timeout.connect(self.showRM65Position)
        self.timer_pos.start(150)  # 设置定时器触发间隔，单位是毫秒
        # 实例化机械臂对象
        self.render_rm65 = self.rm65

        joint[0] = 0
        joint[1] = 0
        joint[2] = 0
        joint[3] = 0
        joint[4] = 0
        joint[5] = 0
        # # MoveJ 运动
        ret = self.rm65.pDll.Movej_Cmd(self.rm65.nSocket, joint, 20, 0, 1)
        print("成功运动到原始位置：")

        joint[0] = -0.421
        joint[1] = 9.728
        joint[2] = 95.016
        joint[3] = 2.34
        joint[4] = 75.415
        joint[5] = -4.5
        # joint[0] = 8.066
        # joint[1] = 7.113
        # joint[2] = 117.814
        # joint[3] = -1.192
        # joint[4] = 57.064
        # joint[5] = -187.550
        ret = self.rm65.pDll.Movej_Cmd(self.rm65.nSocket, joint, 20, 0, 1)
        if ret != 0:
            print("Movec_Cmd 1 失败:" + str(ret))
            sys.exit()
        print("成功运动到待缝合初始位置：")
        print("显示子线程开启")
        self.show_kinect_thread = show_roi_thread(self)
        self.show_kinect_thread.start()
        self.show_kinect_thread.show_status_signal.emit(3)
        # 初始化显示区域，增加一个定时器用作刷新该区域
        print("系统初始化成功，可以点击运行按钮")


    #开始运动按钮 
    def on_runRM65(self):
        # 暂时将图像可视化实时关闭--抬升阶段会影响状态判断的数据输入？
        if(hasattr(self, 'timer_img')):
            self.timer_img.stop()
            print("kinect相机定时器已停止--为了防止状态判断时的数据输入干扰")

        if(not hasattr(self, 'rm65')):
            print("开始run rm65线程")
            rm65 = RobotConnection("192.168.1.19", 8080)
            rm65.connect()
        # 启动机械臂运动线程
        points = []
        with open('data\\points\\6-21\\run.txt', 'r') as file:
            lines = file.readlines()#read data from txt 
            for line in lines:
                points_xyz = [float(val) for val in line.strip().split()]
                # x, y, z = points_xyz[0],points_xyz[1],points_xyz[2]
                # point = DevMsg(x*0.001, y*0.001, z*0.001, -3.117, -0.013, -2.917)
                # cycle point
                x, y, z, rx,ry,rz = points_xyz[0],points_xyz[1],points_xyz[2],points_xyz[3],points_xyz[4],points_xyz[5]
                point = DevMsg(x, y, z, rx, ry, rz)
                points.append(point)

        self.work_thread = WorkThread(self,points)
        self.work_thread.start()


    #更新机械臂位姿信息 
    def showRM65Position(self):
        if hasattr(self, 'render_rm65'):
            cur_pose = self.render_rm65.get_currentPose()
            self.rm65_px.setText(f"{cur_pose.px:.3f}")
            self.rm65_py.setText(f"{cur_pose.py:.3f}")
            self.rm65_pz.setText(f"{cur_pose.pz:.3f}")
            self.rm65_rx.setText(f"{cur_pose.rx:.3f}")
            self.rm65_ry.setText(f"{cur_pose.ry:.3f}")
            self.rm65_rz.setText(f"{cur_pose.rz:.3f}")


    # 机械臂急停
    def on_stopRM65(self):
        print("进入急停--")
        if self.work_thread is not None:
            print("进入急停---急停信号发出")
            self.work_thread.stop_signal.emit()
            self.work_thread = None
    
    # 在自动缝合中暂停该线程
    def on_stopCurrentRM(self):
        # 暂时用于输出机械臂当前位姿
        print("-------------------")
        if(not hasattr(self, 'rm65')):
            self.rm65 = RobotConnection("192.168.1.19", 8080)
            self.rm65.connect()
        cur_point = self.rm65.get_currentPose()
        print(f"{cur_point.px:.6f} {cur_point.py:.6f} {cur_point.pz:.6f} {cur_point.rx:.6f} {cur_point.ry:.6f} {cur_point.rz:.6f}")
        print("暂停按钮--")


    # 开始示教 
    def on_teach(self):
        if(hasattr(self, 'show_kinect_thread')):
            self.show_kinect_thread.stop()
            print("渲染kinect成像子线程停止?")
        # 这里添加示教器模式，对陌生伤口先进行示教，记录其缝合点坐标，保存到本地
        print("success into 示教--修改为收集图像和压力传感器数据入口")
        self.teach_flag = True
        # 如果没有rm65对象，就创建一个
        if(not hasattr(self, 'rm65')):
            self.rm65 = RobotConnection("192.168.1.19", 8080)
            self.rm65.connect()
        # 如果没有kinect对象，就创建一个 
        if(not hasattr(self, 'kinect')):
            self.kinect = KinectCapture()
            if self.kinect.connect() == None:
                print("kinect相机初始化失败")
            print("kinect相机初始化成功")
        # 如果没有压力传感器对象，就创建一个
        if(not hasattr(self,"serial")):
            self.serial=serial.Serial('COM4', 115200)
        self.record_info = "data\\data\\April\\4-24\\1st"
        self.record_info_count = 0
        print("现在可以开始示教了...")
        
    # 示教停止
    def on_stopTeach(self):
        # 停止示教
        self.teach_flag = False
        # 关闭连接
        self.rm65.close()
        self.pressure_thread.stop()
        self.record_info = ""
        self.record_info_count = 0
        print('资源已释放，示教结束')
    
    # 修改为收集第一张图像确定roi区域--待删除废弃
    def on_imgCollect(self):
        print("修改为收集第一张图像确定roi区域")
        if not hasattr(self,'kinect'):
            self.kinect = KinectCapture()
        # 保存点云信息
        file_name = "data\\points\\7-1\\5th\\1st\\"
        # self.kinect.save_point_cloud(filename=file_name)
        # sys.exit()
        # 只有在 KinectCapture 实例存在时才调用相应方法
        colorImg,depthImg = self.kinect.get_frames()
        #  检查colorImg是否为None（即图像是否为空）
        if colorImg is not None:
            # 从四通道的RGBA转换为三通道的RGB
            rgb_img = cv2.cvtColor(colorImg, cv2.COLOR_BGRA2BGR)
            x,y,w,h= self.img_roi[0],self.img_roi[1],self.img_roi[2],self.img_roi[3]
            roi_img = rgb_img[y:y+h, x:x+w].copy()
            # 保存图像，并更新计数
            # file_name="data\\points\\"
            cv2.imwrite(file_name+f"origin_1.png", rgb_img)
            # cv2.imwrite(file_name+img_name, roi_img)
            # predict = predictImg()
            # pred = predict.predict_img(roi_img)
            # cv2.imwrite(file_name+f"pred_{global_count_colorImg}.png", pred)
        print("图像保存成功")
        sys.exit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
