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