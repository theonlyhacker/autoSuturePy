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
        # self.img_roi = [1064,504,330,50] #roi区域,S曲线整个
        # self.img_roi = [1062,512,150,48] #roi区域,s曲线部分c
        # self.img_roi = [1263,380,202,80]#兔子
        self.img_roi = [1020,451,150,40] #roi区域,短粗线部分
        self.run_record_path = "data\\points\\7-15\\3rd\\"

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
        cv2.imwrite(filename + f"\\{self.record_info_count}_High_roi.jpg",roi_img)
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
 
        #修改机械臂工具坐标系 
        current_tool_name = self.rm65.get_currentToolName()
        print("当前工具坐标系:" + str(current_tool_name))
        float_joint = ctypes.c_float * 6
        joint = float_joint()
        str_buf = ctypes.create_string_buffer("autoSu".encode('utf-8'))
        self.rm65.pDll.Change_Tool_Frame(self.rm65.nSocket, str_buf, 1)
        current_tool_name = self.rm65.get_currentToolName()
        print("当前工具坐标系(修改之后):" + str(current_tool_name))

        # 启动定时器，用于实时显示机械臂末端位姿
        self.timer_pos = QTimer(self)
        self.timer_pos.timeout.connect(self.showRM65Position)
        self.timer_pos.start(150)  # 设置定时器触发间隔，单位是毫秒

        joint[0] = 0
        joint[1] = 0
        joint[2] = 0
        joint[3] = 0
        joint[4] = 0
        joint[5] = 0
        # # MoveJ 运动
        ret = self.rm65.pDll.Movej_Cmd(self.rm65.nSocket, joint, 20, 0, 1)
        print("成功运动到原始位置：")
        #初始化相机之后就先提取roi区域，启动相机子线程--保证获取区域不受限制 
        self.show_kinect_thread = show_roi_thread(self)
        self.show_kinect_thread.start()
        self.show_kinect_thread.show_status_signal.emit(3)
        print("显示子线程开启")
        # joint[0] = -0.421
        # joint[1] = 9.728
        # joint[2] = 95.016
        # joint[3] = 2.34
        # joint[4] = 75.415
        # joint[5] = -4.5
        joint[0] = 5.06
        joint[1] = 7.823
        joint[2] = 97.294
        joint[3] = -2.492
        joint[4] = 74.47
        joint[5] = 170.7
        ret = self.rm65.pDll.Movej_Cmd(self.rm65.nSocket, joint, 20, 0, 1)
        if ret != 0:
            print("Movec_Cmd 1 失败:" + str(ret))
            sys.exit()
        print("成功运动到待缝合初始位置：")
        print("系统初始化成功，可以点击运行按钮")
        # 系统所有数据记录路径，将在子线程中路径的定义现在统一交由mian线程控制--集成
        
    #开始运动按钮 
    def on_runRM65(self):
        if(not hasattr(self, 'rm65')):
            print("开始run rm65线程")
            rm65 = RobotConnection("192.168.1.19", 8080)
            rm65.connect()
        # 启动机械臂运动线程
        points = []
        # self.run_record_path = "data\\points\\7-11\\4th\\5th\\"
        with open(self.run_record_path+'plan_data.txt', 'r') as file:
            lines = file.readlines()#read data from txt 
            for line in lines:
                points_xyz = [float(val) for val in line.strip().split()]
                # x, y, z = points_xyz[0],points_xyz[1],points_xyz[2]
                # point = DevMsg(x*0.001, y*0.001, z*0.001, 3.103, 0, -2.879)
                # cycle point
                x, y, z, rx,ry,rz = points_xyz[0],points_xyz[1],points_xyz[2],points_xyz[3],points_xyz[4],points_xyz[5]
                point = DevMsg(x, y, z, rx, ry, rz)
                points.append(point)

        self.work_thread = WorkThread(self,points)
        self.work_thread.start()

    #更新机械臂位姿信息 
    def showRM65Position(self):
        if hasattr(self, 'timer_pos'):
            cur_pose = self.rm65.get_currentPose()
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
    
    # 暂停按钮--暂时用作其他功能-输出当前位姿
    def on_stopCurrentRM(self):
        # 暂时用于输出机械臂当前位姿
        print("-------------------")
        if(not hasattr(self, 'rm65')):
            self.rm65 = RobotConnection("192.168.1.19", 8080)
            self.rm65.connect()
        cur_point = self.rm65.get_currentPose()
        print(f"{cur_point.px:.6f} {cur_point.py:.6f} {cur_point.pz:.6f} {cur_point.rx:.6f} {cur_point.ry:.6f} {cur_point.rz:.6f}")
        # print("暂停按钮--")
        # 通电
        self.rm65.pDll.Get_Tool_Voltage.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_byte))
        self.rm65.pDll.Get_Tool_Voltage.restype = ctypes.c_int
        res = ctypes.c_byte()# 定义一个 ctypes.c_byte 类型的变量
        self.rm65.pDll.Get_Tool_Voltage(self.rm65.nSocket, ctypes.byref(res))# 调用 Get_Tool_Voltage 函数时，传递 res 的地址
        print(f"Current Voltage:  {res.value}")
        self.rm65.pDll.Set_Tool_Voltage.argtypes = (ctypes.c_int, ctypes.c_byte, ctypes.c_bool)
        self.rm65.pDll.Set_Tool_Voltage.restype = ctypes.c_int
        voltage_type = 3 # 设置输入电压为24V 类型为3 # 电源输出类型：0-0V，1-5V，2-12V，3-24V 
        block = True  # 假设需要阻塞执行
        set_voltage_result = self.rm65.pDll.Set_Tool_Voltage(self.rm65.nSocket, voltage_type, block)
        print(f"Set Voltage Result: {set_voltage_result}")
        # 查看当前电压
        current_voltage = ctypes.c_byte()
        get_voltage_result = self.rm65.pDll.Get_Tool_Voltage(self.rm65.nSocket, ctypes.byref(current_voltage))
        print(f"Get Voltage Result: {get_voltage_result}")
        print(f"Current Voltage: {current_voltage.value}")
        # 通信
        # ret = self.rm65.pDll.Set_Modbus_Mode(self.rm65.nSocket,1,115200,1000,1)

    # 开始示教 
    def on_teach(self):
        if(hasattr(self, 'show_kinect_thread')):
            self.show_kinect_thread.stop()
            print("渲染kinect成像子线程停止?")
        # 这里添加示教器模式，对陌生伤口先进行示教，记录其缝合点坐标，保存到本地
        print("success into 示教--修改为收集图像和压力传感器数据入口")
        self.teach_flag = True
        # 如果没有rm65对象，就创建一个
        # if(not hasattr(self, 'rm65')):
        #     self.rm65 = RobotConnection("192.168.1.19", 8080)
        #     self.rm65.connect()
        # 如果没有kinect对象，就创建一个 
        if(not hasattr(self, 'kinect')):
            self.kinect = KinectCapture()
            if self.kinect.connect() == None:
                print("kinect相机初始化失败")
            print("kinect相机初始化成功")
        # 如果没有压力传感器对象，就创建一个
        if(not hasattr(self,"serial")):
            self.serial=serial.Serial('COM4', 115200)
        file_name = self.run_record_path+"teach"
        os.makedirs(os.path.join(file_name), exist_ok=True)
        self.record_info = file_name
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
    
    # 示教前图像信息采集
    def on_imgCollect(self):
        # # 测试压力传感器是否能用
        # self.img_thread = collect_Kinect_Pressure(self)
        # self.img_thread.start()
        # time.sleep(3)
        # exit()
        cur_img_num = 0
        print("修改为收集图像以及确定roi区域")
        if not hasattr(self,'kinect'):
            self.kinect = KinectCapture()
        # 图像信息
        # self.kinect.save_point_cloud("data\\points\\7-15\\cloud_data\\3rd\\")
        # print("点云保存结束")
        # file_name = "data\\points\\7-15\\origin_img\\3rd\\"
        file_name = self.run_record_path +"origin"
        os.makedirs(os.path.join(file_name), exist_ok=True)
        for i in range(30):
            colorImg,depthImg = self.kinect.get_frames()
            #  检查colorImg是否为None（即图像是否为空）
            if colorImg is not None:
                # 从四通道的RGBA转换为三通道的BGR
                rgb_img = cv2.cvtColor(colorImg, cv2.COLOR_BGRA2BGR)
                # print(file_name)
                cv2.imwrite(file_name+f"\\origin_{i}.png", rgb_img)
        print("图像收集成功")
        sys.exit()
        # cur = 0
        # path="data\\points\\7-15\\rec\\"
        # while(1):
        #     colorImg,depthImg = self.kinect.get_frames()
        #     if colorImg is not None:
        #         cv2.imwrite(path+f"{cur}.jpg",colorImg)
        #         cur = cur+1
        #         time.sleep(0.04)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
