"""
General GridWorld Environment
"""
import math
import gym
from gym import spaces
from gym.utils import seeding
import cv2
import numpy as np
import copy
import os
import torch

# 一些注意事项：step里的reward必须是负数，目前不清楚为什么
# 定义格子的属性：坐标、是否是障碍物、即时奖励、value
class Grid(object):
    def __init__(self, x: int = None,
                 y: int = None,
                 type: int = 0,
                 reward: float = 0.0,
                 value: float = 0.0):  # value属性备用
        self.x = x  # 坐标x
        self.y = y
        self.type = type  # 类别值（0：空；1：障碍或边界）
        self.reward = reward  # 该格子的即时奖励
        self.value = value  # 该格子的价值，暂没用上
        self.name = None  # 该格子的名称
        self._update_name()

    def _update_name(self):
        self.name = "X{0}-Y{1}".format(self.x, self.y)

    def __str__(self):
        return "name:{4}, x:{0}, y:{1}, type:{2}, value:{3}".format(self.x,
                                                                    self.y,
                                                                    self.type,
                                                                    self.value,
                                                                    self.name
                                                                    )

# 格子矩阵，用来定义世界，对伤口环境的建模需要使用这个部分
class GridMatrix(object):
    '''格子矩阵，通过不同的设置，模拟不同的格子世界环境
    '''

    def __init__(self, n_width: int,  # 水平方向格子数
                 n_height: int,  # 竖直方向格子数
                 default_type: int = 0,  # 默认类型
                 default_reward: float = 0.0,  # 默认即时奖励值
                 default_value: float = 0.0  # 默认价值（这个有点多余）
                 ):
        self.grids = None
        self.n_height = n_height
        self.n_width = n_width
        self.len = n_width * n_height
        self.default_reward = default_reward
        self.default_value = default_value
        self.default_type = default_type
        self.reset()
        self.distance = 0
    # 重启环境，所有的格子默认r v为0
    def reset(self, random: bool = False):
        random = True
        if not random:
            np.random.seed(0)
        self.grids = []
        self.path = []

        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x,
                                       y,
                                       self.default_type,
                                       self.default_reward,
                                       self.default_value))

    def get_grid(self, x, y=None):
        '''获取一个格子信息
        args:坐标信息，由x，y表示或仅有一个类型为tuple的x表示
        return:grid object
        '''
        xx, yy = None, None
        if isinstance(x, int):
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], x[1]
        # print(xx, yy)
        assert (xx >= 0 and yy >= 0 and xx < self.n_width and yy < self.n_height), "任意坐标值应在合理区间"
        index = yy * self.n_width + xx
        return self.grids[index]

    def set_reward(self, x, y, reward):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.reward = reward
        else:
            raise ("grid doesn't exist")

    def set_value(self, x, y, value):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.value = value
        else:
            raise ("grid doesn't exist")

    def set_type(self, x, y, type):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.type = type
        else:
            raise ("grid doesn't exist")

    def get_reward(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.reward

    def get_value(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.value

    def get_type(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.type

class GridWorldEnv(gym.Env):
    '''格子世界环境，可以模拟各种不同的格子世界
    '''
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, n_width: int = 14,
                 n_height: int = 50,
                 u_size=12,
                 default_reward: float = -1,
                 default_type=0,
                 windy=False,
                 img_path = "fillwound.png",
                 up = (758, 320),   #758,320, 912-758, 352-320
                 down = (912, 352)
                 ):
        # self.display = display
        self.u_size = u_size  # 当前格子绘制尺寸
        self.n_width = n_width  # 格子世界宽度（以格子数计）
        self.n_height = n_height  # 高度
        self.up = up
        self.down = down
        self.default_reward = default_reward
        self.default_type = default_type
        self._adjust_size()
        self.img_path = img_path
        # grids_number, gray_image = getImage("Swound.png")
        grids_number, gray_image = getImage(self.img_path, self.up, self.down)
        # print(grids_number)
        self.n_width = grids_number[1]
        self.n_height = grids_number[0]
        self.u_size=int(122*5/grids_number[0])
        self.grids = GridMatrix(n_width=self.n_width,
                                n_height=self.n_height,
                                default_reward=self.default_reward,
                                default_type=self.default_type,
                                default_value=0.0)
        self.reward = 0  # for rendering
        self.action = None  # for rendering
        # 无风世界
        self.windy = False  # 是否是有风格子世界
        # 设置观测空间和动作空间
        self.low_bounds = np.array([-self.n_width/2, -10, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)  # 每个维度的下界，确保是整数
        self.high_bounds = np.array([self.n_width/2, self.n_height, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32)  # 每个维度的上界，确保是整数
        # 创建一个Box空间，其中每个维度都有自己的整数界限 
        self.observation_space = spaces.Box(low=self.low_bounds, high=self.high_bounds, dtype=np.int32)
        self.action_space = spaces.Discrete(8)
        self.distance = 0
        self.ends = [(1, 1)]  # 终止格子坐标，可以有多个
        self.start = (1, 1)  # 起始格子坐标，只有一个
        self.types = []  # 特殊种类的格子在此设置。[(3,2,1)]表示(3,2)处值为1
        self.origin_types = []
        self.rewards = []  # 特殊奖励的格子在此设置，终止格子奖励0
        self.last_point_x = 0
        self.last_point_y = 0
        self.this_point_x = 0
        self.this_point_y = 0
        self.path = []
        self.chuzhenP = []
        self.point_S = []
        self.wound_center = []
        self.wound_width = []
        self.origin_wWidth = []
        self.count = 0
        self.total_step = 200
        self.flag = 1
        self.line_width = 3
        self.refresh_setting()
        self.viewer = None  # 图形接口对象
        self.seed()  # 产生一个随机子
        self.reset()
        
        
        # print(self.n_width, self.n_height)
        self.width = self.u_size * self.n_width  # 场景宽度 screen width
        self.height = self.u_size * self.n_height  # 场景长度
        self.line_width = 4
        self.distance = 3
        self.wound_center, self.origin_wWidth, self.origin_types, self.start, self.ends[0] = getTypes(gray_image, grids_number)
        self.types = copy.deepcopy(self.origin_types)
        self.wound_width = copy.deepcopy(self.origin_wWidth)
        # print(self.wound_width)
        self.refresh_types() 
        self.total_sum = 0
        # print("start end:", self.start, self.ends)
        # self.start = (6, 4)
        # self.ends = [(6, 46)]
        # self.position = (0, 0)

    def init(self, display=False):
        self.display = display
        return self


    # 最大宽度、高度不超过800，可以考虑多画格子，等代码搭好修改变量尝试
    def _adjust_size(self):
        '''调整场景尺寸适合最大宽度、高度不超过800
        '''
        pass

    def seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # 这里是重头戏，要好好看好好学！！！！reward的设置需要在这里处理
    def step(self, action):
        self.count += 1
        self.path.append(self.position)
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.action = action  # action for rendering
        pos_x, pos_y = self.position
        old_x, old_y = self.state[0], self.state[1]
        new_x, new_y = old_x, old_y
        old_p1, old_p2, old_p3, old_p4, old_p5, old_p6, old_p7, old_p8 = self.state[2], self.state[3],self.state[4], self.state[5], self.state[6], self.state[7], self.state[8], self.state[9]
        new_p1, new_p2, new_p3, new_p4, new_p5, new_p6, new_p7, new_p8 = old_p1, old_p2, old_p3, old_p4, old_p5, old_p6, old_p7, old_p8
        
        if 0 <= action < 1:
            action = 0
            new_x += 1  
            pos_x -= 1# left
        elif 1 <= action < 2:
            action = 1
            new_x -= 1  
            pos_x += 1# right
        elif 2 <= action < 3:
            action = 2
            new_y -= 1
            pos_y += 1  # up
        elif 3 <= action < 4:
            action = 3
            new_y += 1  
            pos_y -= 1# down
        elif 4 <= action < 5: # 左下
            action = 4
            new_x, new_y = new_x + 1, new_y + 1
            pos_x, pos_y = pos_x - 1, pos_y - 1
        elif 5 <= action < 6: # 右下
            action = 5
            new_x, new_y = new_x - 1, new_y + 1
            pos_x, pos_y = pos_x + 1, pos_y - 1
        elif 6 <= action < 7: # 左上
            action = 6
            new_x, new_y = new_x + 1, new_y - 1
            pos_x, pos_y = pos_x - 1, pos_y + 1
        elif 7 <= action < 8: # 右上
            action = 7
            new_x, new_y = new_x - 1, new_y - 1
            pos_x, pos_y = pos_x + 1, pos_y + 1
        # boundary effect
        if pos_x < 1 or pos_x >= int(self.n_width-1) or pos_y < 1 or pos_y >= self.n_height-2 or self.grids.get_type(pos_x, pos_y) == 1:
            new_x, new_y = old_x, old_y
            pos_x, pos_y = self.position[0], self.position[1]
        else:
            self.position = (pos_x, pos_y)

            
        # 修改points/修改dis
        if 0 <= action <= 7:
            new_p1 = self.grids.get_type(pos_x-1, pos_y+1)
            new_p2 = self.grids.get_type(pos_x, pos_y+1)
            new_p3 = self.grids.get_type(pos_x+1, pos_y+1)
            new_p4 = self.grids.get_type(pos_x-1, pos_y)
            new_p5 = self.grids.get_type(pos_x+1, pos_y)
            new_p6 = self.grids.get_type(pos_x-1, pos_y-1)
            new_p7 = self.grids.get_type(pos_x, pos_y-1)
            new_p8 = self.grids.get_type(pos_x+1, pos_y-1)
            # 修改distance
            # new_dis = self.wound_center[new_y] - new_x
        # self.path.append((new_x, new_y))

        # 距离reward
        if action > 3 and action < 8:  # 斜向走
            self.reward = -1
        elif action < 4:
            self.reward = -1
        self.reward -= 0.4*(abs((self.wound_center[pos_y] - self.wound_width[pos_y]/2- pos_x))/2)

        done = False
        info = {"Finding"}
        # 如果到终点，计算最终的reward
        if new_x == 0 and new_y == 0:
            done_r = 100
            done = True
            self.reward = done_r # + suture_r #- self.total_sum
            info = {"Arrive!"}
        self.grids.set_reward(pos_x, pos_y, self.reward)
        self.state = (new_x, new_y, new_p1, new_p2, new_p3, new_p4, new_p5, new_p6, new_p7, new_p8)
        # 提供格子世界所有的信息在info内
        done = done or self.count >= self.total_step
        return self.state, self.reward, done, info


    # 将状态变为横纵坐标。注意这里的状态，这里需要进一步考虑，是不是存在降维的操作
    # 命名保护
    def _state_to_xy(self, s):
        x = s % self.n_width
        y = int((s - x) / self.n_width)
        return x, y

    def _xy_to_state(self, x, y=None):
        if isinstance(x, int):
            assert (isinstance(y, int)), "incomplete Position info"
            return x + self.n_width * y
        elif isinstance(x, tuple):
            return x[0] + self.n_width * x[1]
        return -1  # 未知状态

    def refresh_setting(self):
        '''用户在使用该类创建格子世界后可能会修改格子世界某些格子类型或奖励值
        的设置，修改设置后通过调用该方法使得设置生效。
        '''
        for x, y, r in self.rewards:
            self.grids.set_reward(x, y, r)
        for x, y, t in self.types:
            self.grids.set_type(x, y, t)

    def refresh_types(self):
        '''用户在使用该类创建格子世界后可能会修改格子世界某些格子类型或奖励值
        的设置，修改设置后通过调用该方法使得设置生效。
        '''
        for x, y, t in self.types:
            self.grids.set_type(x, y, t)

    def reset(self, random: bool = False):
        self.path = []
        self.types = copy.deepcopy(self.origin_types)
        self.wound_width = copy.deepcopy(self.origin_wWidth)
        self.refresh_types() 
        # 最后一维是0,意味着处理的是直线型，仅在reset的部分，选择处理的图像。
        self.chuzhenP = []
        # # Lwound
        # self.state = (0, 42, 0,0,0,0,0,0,0,0)
        # self.position = (6,4)
        # self.start = (6, 4)
        # self.ends = [(6, 46)]

        # Swound
        # self.start = (11, 4)
        # self.ends = [(9, 52)]
        # self.position = (11, 4)
        # self.state = (-2, 48, 0,0,0,0,0,0,0,0)

        # multi-wound
        self.state = (self.ends[0][0]-self.start[0], self.ends[0][1]-self.start[1], 0,0,1,0,1,0,0,0)
        self.position = (self.start[0],self.start[1])

        self.count = 0
        self.flag = 1
        return self.state
        # return torch.as_tensor(self.state, dtype=torch.int)

    # 判断是否是终止状态
    def _is_end_state(self, x, y=None):
        if y is not None:
            xx, yy = x, y
        elif isinstance(x, int):
            xx, yy = self._state_to_xy(x)
        else:
            assert (isinstance(x, tuple)), "坐标数据不完整"
            xx, yy = x[0], x[1]
        for end in self.ends:
            if xx == end[0] and yy == end[1]:
                return True
        return False

    def get_path(self, path):
        self.path = path

    # 图形化界面
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        zero = (0, 0)
        u_size = self.u_size
        m = 1  # 格子之间的间隙尺寸
        from gym.envs.classic_control import rendering
        # 如果还没有设定屏幕对象，则初始化整个屏幕具备的元素。
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.width, self.height)
        # print(self.path)
        # 绘制格子
        for x in range(self.n_width):
            for y in range(self.n_height):
                v = [(x * u_size + m, y * u_size + m),
                     ((x + 1) * u_size - m, y * u_size + m),
                     ((x + 1) * u_size - m, (y + 1) * u_size - m),
                     (x * u_size + m, (y + 1) * u_size - m)]
                rect = rendering.FilledPolygon(v)
                rect.set_color(0.9, 0.6, 0.6)
                self.viewer.add_geom(rect)
                # 绘制边框
                v_outline = [(x * u_size + m, y * u_size + m),
                             ((x + 1) * u_size - m, y * u_size + m),
                             ((x + 1) * u_size - m, (y + 1) * u_size - m),
                             (x * u_size + m, (y + 1) * u_size - m)]
                outline = rendering.make_polygon(v_outline, False)
                outline.set_linewidth(3)
                if self._is_end_state(x, y):
                    # 给终点方格添加金黄色边框
                    outline.set_color(0.9, 0.9, 0)
                    self.viewer.add_geom(outline)
                # 绘制起点
                if self.start[0] == x and self.start[1] == y:
                    outline.set_color(0.5, 0.5, 0.8)
                    self.viewer.add_geom(outline)
                if self.grids.get_type(x, y) == 1:  # 障碍格子用深灰色表示
                    rect.set_color(0.3, 0.3, 0.3)
                else:
                    pass
            # 绘制个体
            self.agent = rendering.make_circle(u_size / 4, 30, True)
            self.agent.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)
            # 更新个体位置
        for i in range(len(self.path)):
            x, y = (self.path[i])
            # x = x + int(self.wound_width[]/2)
            if i==len(self.path)-1:
                # next_x, next_y = (self.ends[0][0], self.ends[0][1])
                # next_x, next_y = x,y
                next_x, next_y = self.ends[0][0]-self.state[0], self.ends[0][1]-self.state[1]
            else:
                next_x, next_y = self.path[i+1]
            if self.path[i] in self.chuzhenP:
                self.aline = rendering.Line(((x + 0.5) * u_size, (y + 0.5) * u_size),
                                            ((2*self.wound_center[next_y] - x + 0.5) * u_size, (y + 0.5) * u_size))
                self.viewer.add_geom(self.aline)
                self.aline = rendering.Line(((2*self.wound_center[next_y] - x + 0.5) * u_size, (y + 0.5) * u_size),
                                            ((x + 0.5) * u_size, (y + 0.5) * u_size))
                self.viewer.add_geom(self.aline)
                self.aline = rendering.Line(((x + 0.5) * u_size, (y + 0.5) * u_size),
                                            ((next_x + 0.5) * u_size, (next_y + 0.5) * u_size))
                self.viewer.add_geom(self.aline)
            else:
                self.aline = rendering.Line(((x + 0.5) * u_size, (y + 0.5) * u_size),
                                            ((next_x + 0.5) * u_size, (next_y + 0.5) * u_size))
                self.viewer.add_geom(self.aline)
            self.aline.set_color(100, -100, -100)
        # self.viewer.add_geom(self.aline)
        # 绘制伤口中轴线
        for i in range(1, len(self.wound_center)):
            if not(self.wound_center[i] == 0) and not(self.wound_center[i-1] == 0):
                self.aline = rendering.Line(((self.wound_center[i-1]+ 0.5) * u_size,(i - 1 + 0.5) * u_size),
                                            ((self.wound_center[i]+ 0.5) * u_size,(i + 0.5) * u_size))
                # 把图形元素添加到画板中
                self.viewer.add_geom(self.aline)

        x, y = self.ends[0][0]-self.state[0], self.ends[0][1]-self.state[1]
        self.agent_trans.set_translation((x + 0.5) * u_size, (y + 0.5) * u_size)
        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

def get_demo(path="", n_demo=20, display=False):
    if os.path.isfile(path):
        print(f"Demo Loaded from {path}")    
        sample= torch.load(path)
        # sample = dataNorm(sample)
        return sample,n_demo
    else:
        raise KeyError(f"the data path must exist")

def get_path_length(path, width):
    length = 0
    for i in range(len(path) - 1):
        if abs(path[i] - path[i + 1]) == 1:
            length += 1
        elif abs(path[i] - path[i + 1]) == width:
            length += 1
        else:
            length += 1.4
    return length

# 用于保存最终运行结果GIF的方法
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def display_frames_as_gif(frames):
    # 创建一个新的figure对象，并设置其大小
    fig, ax = plt.subplots(figsize=(2, 6))  # 这里的figsize参数设置了画布的大小
    # 清除坐标轴
    ax.axis('off')
    # 初始化图像显示
    patch = ax.imshow(frames[0], animated=True)  # 添加animated=True以确保动画有效

    def animate(i):
        patch.set_data(frames[i])
        # 使用fig（而不是plt.gcf()）作为动画的figure
    anim = FuncAnimation(fig, animate, frames=range(len(frames)),
                         interval=50)  # 注意这里frames参数通常是一个可迭代对象，这里使用range(len(frames))
    # 保存GIF动画
    anim.save('./result.gif', writer='pillow', fps=10)

# 得到尺寸用于初始化env
def getImage(imgpath, upP, downP):
    world_image = cv2.imread(imgpath)
    gray_image = cv2.cvtColor(world_image, cv2.COLOR_BGR2GRAY)
    x, y, width, height = upP[0], upP[1], downP[0]-upP[0], downP[1]-upP[1]
    # 裁切图像  
    cropped_image = gray_image[y:y+height, x:x+width]  
    # cropped_image = cv2.flip(cv2.transpose(cropped_image), 1)  
    # 获取图像的高度和宽度  
    h, w = cropped_image.shape[:2]  
    # 创建一个新的图像来保存旋转后的结果，注意宽度和高度要互换  
    rotated_image = np.zeros((w, h), dtype=cropped_image.dtype)  
    # 逆时针旋转90度，重新排列像素  
    for y in range(h):  
        for x in range(w):  
            rotated_image[x, h-y-1] = cropped_image[y, x]  # 注意y坐标的调整  
    cv2.imwrite("grayimage.png", rotated_image)
    # gray_image = cv2.cvtColor(world_image, cv2.COLOR_BGR2GRAY)
    # cropped_image = cv2.flip(cv2.transpose(cropped_image), 1)  
    grid_height = 3
    grid_width = 3
    height, width = rotated_image.shape
    # print("image shape:",height, width)
    grid_size = (int(height / grid_height), int(width / grid_width))
    # print(grid_size)
    return grid_size, rotated_image

# 封装一个读取图片的函数，根据图片得到types，之后直接赋值给env.types
def getTypes(gray_image, grid_size):
    wound_width = []
    wound_center = []
    grids_types = []
    start_on = 1
    end_on = 1
    startP = (1, 1)
    endP = (1, 1)
    endY = 0
    grid_height = 3
    grid_width = 3
    all_grid_height = grid_height * grid_size[0]
    all_grid_width = grid_width * grid_size[1]
    gray_image = gray_image[:all_grid_height, :all_grid_width]
    # binary_image = np.zeros_like(gray_image)
    # 伤口描述：
    # 伤口中线（用作伤口缝合完毕的展示环节）、伤口宽度（用于伤口收缩）
    # 遍历每个网格单元
    # print("grid_size:", grid_size[1], grid_size[0])    grid_size[1], grid_size[0],分别对应xy
    for i in range(grid_size[0]):
        wound_row = []
        wound_w = 0
        # startP = (1, 1)
        change_signal = 0
        for j in range(grid_size[1]):
            # 计算网格单元的坐标
            x_start = j * grid_width
            x_end = x_start + grid_width
            y_start = i * grid_height
            y_end = y_start + grid_height

            # 获取网格中心点的坐标
            center_x = x_start + grid_width // 2
            center_y = y_start + grid_height // 2

            # 获取网格中心的颜色值
            center_value = gray_image[center_y, center_x]

            # 设置阈值
            thresh = 166  # 可以根据需要调整这个值

            # 根据中心点的颜色值对整个网格单元进行二值化
            if center_value < thresh:
                change_signal = 1
                if start_on and j>2 and i>0:
                    startP = (j-1, i)
                    start_on = 0
                
                wound_w += 1
                grids_types.append((j, i, 1))
                wound_row.append(j)
                endY = endY if endY > i else i
            else:
                grids_types.append((j, i, 0))
        if change_signal == 0 and end_on and (not start_on):
            endP = (int(wound_center[i-1]-wound_width[i-1]/2), i-1)
            end_on = 0
        if i==grid_size[0]-1 and end_on:
            # 设置终点
            endP = (int(wound_center[i-2]-wound_width[i-2]/2-1), i-2)
            end_on = 0
            # 使得最顶行为伤口外
            for top in range(grid_size[1]):
                grids_types.append((top, i, 0))
                grids_types.append((top, i-1, 0))
        wound_width.append(wound_w)
        if not wound_row :
            wound_center.append(0)
        else:
            wound_center.append(np.ceil(np.average(wound_row))) # 向上取整，更有利于保证伤口完全缝合。但是也可以根据实际情况选择更好描述伤口中线的那一个
    return wound_center, wound_width, grids_types, startP, endP