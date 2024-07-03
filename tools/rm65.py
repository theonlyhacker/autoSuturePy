import ctypes
import os
import sys
#   调用结构体 POSE
class POSE(ctypes.Structure):
    _fields_ = [("px", ctypes.c_float),
                ("py", ctypes.c_float),
                ("pz", ctypes.c_float),
                ("rx", ctypes.c_float),
                ("ry", ctypes.c_float),
                ("rz", ctypes.c_float)]
#   调用结构体 POSE 
class DevMsge(ctypes.Structure):
    _fields_ = [("frame_name", ctypes.c_char * 10),
                ("pose", POSE),
                ("payload", ctypes.c_float),
                ("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("z", ctypes.c_float)]
    

class RobotConnection:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.nSocket = None
        self.pDll = None

    def connect(self):
        CUR_PATH = os.path.dirname(os.path.realpath(__file__))
        dllPath = os.path.join(CUR_PATH, "RM_Base.dll")
        self.pDll = ctypes.cdll.LoadLibrary(dllPath)

        # API 初始化
        self.pDll.RM_API_Init(65, 0)

        # 连接机械臂
        byteIP = bytes(self.ip, "gbk")
        self.nSocket = self.pDll.Arm_Socket_Start(byteIP, self.port, 200)
        print("机械臂连接成功 m_sockhand:",self.nSocket)

    # 获取当前工具坐标系名字
    def get_currentToolName(self):
        self.pDll.Get_Current_Tool_Frame.argtypes = (ctypes.c_int, ctypes.POINTER(DevMsge))
        self.pDll.Get_Current_Tool_Frame.restype = ctypes.c_int
        tool_farme = DevMsge()
        ret = self.pDll.Get_Current_Tool_Frame(self.nSocket, tool_farme)
        # print("当前工具坐标系:" + str(tool_farme.frame_name))
        return tool_farme.frame_name

    # 获取当前坐标系末端位姿Pose
    def get_currentPose(self):
        self.pDll.Get_Current_Arm_State.argtypes = (ctypes.c_int, ctypes.c_float * 6, ctypes.POINTER(POSE), ctypes.c_uint16 * 6, ctypes.c_uint16 * 6)
        self.pDll.Get_Current_Arm_State.restype = ctypes.c_int
        float_joint = ctypes.c_float * 6
        joint1 = float_joint()
        pose = POSE()
        arm_err = (ctypes.c_uint16 * 6)()
        sys_err = (ctypes.c_uint16 * 6)()
        ret  = self.pDll.Get_Current_Arm_State(self.nSocket, joint1, pose, arm_err, sys_err)
        # print("the ret is :", ret)
        # print("the arm_err is :", arm_err, "the sys_err is :", sys_err)
        # print("当前关节角度:" + str(joint1[0])+"   "+str(joint1[1])+"    "+str(joint1[2])+"    "+str(joint1[3])+"    "+str(joint1[4])+"    "+str(joint1[5]))
        # print("当前坐标点信息:" + str(pose.px)+"   "+str(pose.py)+"    "+str(pose.pz))
        # self.pDll.Get_Joint_Degree.argtypes = (ctypes.c_int, ctypes.c_float * 6)
        # self.pDll.Get_Joint_Degree.restype = ctypes.c_int
        # joint2 = float_joint()
        # ret = self.pDll.Get_Joint_Degree(self.nSocket,joint2)
        # print("the joint's ret is ", ret)
        # print("当前关节角度:" + str(joint2[0])+"   "+str(joint2[1])+"    "+str(joint2[2])+"    "+str(joint2[3])+"    "+str(joint2[4])+"    "+str(joint2[5]))

        return pose
    

    def record_run(self):
        # 获取当前坐标系
        tool_name = self.get_currentToolName()
        print("111---111" + str(tool_name))
        if(tool_name != b'suture'):
            sys.exit("当前坐标系不是suture")
        # print("当前坐标点信息:" + str(tool_farme.pose))
        
        # return tool_farme.pose.px, tool_farme.pose.py, tool_farme.pose.pz, tool_farme.pose.rx, tool_farme.pose.ry, tool_farme.pose.rz
    
    def run_test(self):
        tool_name = rm65.get_currentToolName()
        print("run_test------" + str(tool_name))
        if(tool_name != b'suture'):
            sys.exit("当前坐标系不是suture")
        #   初始位置
        float_joint = ctypes.c_float*6
        joint = float_joint()

        joint[0] = 0
        joint[1] = 0
        joint[2] = 0
        joint[3] = 0
        joint[4] = 0
        joint[5] = 0
        # # MoveJ 运动
        ret = rm65.pDll.Movej_Cmd(rm65.nSocket, joint, 20, 0, 1)
        if ret != 0:
            print("Movej_Cmd 1 失败:" + str(ret))
            sys.exit()

        joint[0] = 8.305
        joint[1] = 13.450
        joint[2] = 109.871
        joint[3] = -1.293
        joint[4] = 58.724
        joint[5] = -187.275
        ret = rm65.pDll.Movej_Cmd(rm65.nSocket, joint, 20, 0, 1)

        # po4 = DevMsg()
        # po4.px = -0.252160
        # po4.py = -0.0157
        # po4.pz = 0.163
        # po4.rx = 3.141
        # po4.ry = -0.01
        # po4.rz = 1.024
        # rm65.pDll.Movel_Cmd.argtypes = (ctypes.c_int, DevMsg, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)
        # rm65.pDll.Movel_Cmd.restype = ctypes.c_int
        # ret = rm65.pDll.Movel_Cmd(rm65.nSocket, po4, 20, 0, 1)
        # if ret != 0 :
        #     print("Movel_Cmd 2 失败:" + str(ret))
        # sys.exit()

        points = []
        with open('test_only.txt', 'r') as file:
            for line in file:
                x, y, z = map(float, line.split())
                print(x,y,z)
                # point = DevMsg(x*0.001, y*0.001, z*0.001, 3.141, -0.01, 1.024)
                point = POSE(x*0.001, y*0.001, z*0.001, -3.118, -0.0013, -2.917)
                points.append(point)
        # sys.exit(0)
        for index, point in enumerate(points):
            # 你的代码
            # for point ,index in points:
            # temp_point = DevMsg()
            # rm65.pDll.Movel_Cmd.argtypes = (ctypes.c_int, DevMsg, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)
            # rm65.pDll.Movel_Cmd.restype = ctypes.c_int
            print("index is :", index)
            print("the point.pz is :", point.pz)
            # 向下运动到待缝合表面
            ret = rm65.pDll.Movel_Cmd(rm65.nSocket, point, 20, 0, 1)
            if ret != 0 :
                print("Movel_Cmd 失败:" + str(ret))
            
            # 缝合动作区间
            # 。。。
            # 缝合完成向上提取,快速区间
            temp_point = point
            temp_point.pz = (25 - 2*index)*0.01 + temp_point.pz
            print("the temp_point.pz is :", temp_point.pz)
            ret = rm65.pDll.Movel_Cmd(rm65.nSocket, temp_point, 50, 0, 1)
            if ret != 0 :
                print("Movel_Cmd 失败:" + str(ret))


    def close(self):
        # 关闭连接
        self.pDll.Arm_Socket_Close(self.nSocket)
    

if __name__ == "__main__":
    # 实例化RM65类
    rm65 = RobotConnection("192.168.1.19", 8080)
    # 初始化实例变量
    rm65.connect()
    # 获取当前坐标系
    temp_pose = rm65.get_currentPose()
    print("the temp_pose is :", temp_pose.px, temp_pose.py, temp_pose.pz, temp_pose.rx, temp_pose.ry, temp_pose.rz)
    # 关闭socket
    rm65.pDll.Arm_Socket_Close(rm65.nSocket)