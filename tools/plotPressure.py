import matplotlib.pyplot as plt
import glob
import os

# 获取文件夹下所有的txt文件dataPressure/11-18/1th/signal
files = glob.glob('dataPressure/1-19/7th/**/*.txt')
def plot_pressure_data_old(files):
    # 遍历每个文件
    for file in files:
        # 读取文件内容
        with open(file, 'r') as f:
            data = [float(line.strip()) for line in f]
        # 创建x坐标
        x = list(range(len(data)))
        # 绘制折线图
        plt.plot(x, data)
        # 根据文件名保存图片
        plt.savefig(file.replace('.txt', '.png'))
        # 清空当前图像
        plt.clf()


def plot_pressure_data_allTime(folder_path):
    # 循环遍历每个子文件夹
    pressure_file = os.path.join(folder_path, 'pressure_all_time.txt')
    # 读取数据并分别存储数值
    values = []
    with open(pressure_file, 'r') as f:
        for line_num, line in enumerate(f, 1):  # 从1开始计数行号
            value = line.strip().split()[1]  # 获取第二个数值
            values.append(float(value))  # 将数值转换为浮点数

    # 绘制图形
    plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(values) + 1), values, marker='o', linestyle='-')
    plt.scatter(range(1, len(values) + 1), values)
    plt.xlabel('Line Number')
    plt.ylabel('Value')
    plt.title('Data Visualization')
    plt.grid(True)
    plt.show()

def plot_pressure_data(folder_path):
    # 获取文件夹下的所有子文件夹
    # subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    # 创建一个图形窗口
    plt.figure(figsize=(10, 6))

    # 循环遍历每个子文件夹
    # for folder in subfolders:
    pressure_file = os.path.join(folder_path, 'pressure_all.txt')
    if os.path.exists(pressure_file):
        # 读取压力数据
        with open(pressure_file, 'r') as f:
            pressures = [float(line.strip()) for line in f.readlines()]
        
        # 绘制压力数据
        plt.plot(pressures, label=os.path.basename(folder_path))

    # 添加标题和标签
    plt.title('Pressure Data')
    plt.xlabel('Time')
    plt.ylabel('Pressure')
    plt.legend()  # 显示图例

    # 显示图形 wound_points_3d
    plt.show()

def plot_wound_edge_data(folder_path):
    # 获取文件夹下的所有子文件夹
    # subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    subfolders = ["data\\data\\3-14\\1st\\suture_0",
                  "data\\data\\3-14\\1st\\suture_4",
                  "data\\data\\3-14\\1st\\suture_7"]
    # 创建一个3D图形窗口
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 循环遍历每个子文件夹
    for folder in subfolders:
        pressure_file = os.path.join(folder, 'wound_points_3d.txt')
        if os.path.exists(pressure_file):
            # 读取压力数据
            with open(pressure_file, 'r') as f:
                lines = f.readlines()
                x_values = []
                y_values = []
                z_values = []
                for line in lines:
                    # 将每行数据以空格分割，并转换为浮点数
                    pressure_values = [float(val) for val in line.strip().split()]
                    x_values.append(pressure_values[0])
                    y_values.append(pressure_values[1])
                    z_values.append(pressure_values[2])
            
            # 绘制3D折线图
            # ax.plot(x_values, y_values, z_values, label=os.path.basename(folder))
            # 绘制散点图
            ax.scatter(x_values, y_values, z_values, label=os.path.basename(folder))

    # 添加标题和标签
    ax.set_title('Wound Edge Data (3D)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()  # 显示图例

    # 显示图形
    plt.show()

def plot_3_20_pressure_data(folder_path):
    # 存储所有文件中的数据
    all_data = []

    # 统计空文件个数
    empty_files_count = 0

    # 遍历文件夹中的每个txt文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                data = file.readlines()  # 读取文件中的所有行数据
                data = [line.strip() for line in data]  # 去除每行末尾的换行符
                if len(data) > 0:  # 如果文件不为空
                    all_data.append(data)  # 将数据存储到列表中
                else:
                    empty_files_count += 1  # 空文件计数加1

    # 统计并输出空文件个数
    print("Number of empty files:", empty_files_count)

    # 绘制数据
    plt.figure(figsize=(10, 6))
    for data in all_data:
        x = range(1, len(data) + 1)  # X轴为数据行数
        y = [float(value) for value in data]  # 将数据转换为浮点数
        plt.scatter(x, y)  # 绘制散点图
    plt.xlabel('Line Number')
    plt.ylabel('Value')
    plt.title('Data Visualization')
    plt.grid(True)
    plt.show()
    
def plot_top_pos(path):
    x = []
    y = []
    z = []
    with open(path, 'r') as file:
        for i, line in enumerate(file):
            if i < 3:  # 跳过前三个数据点
                continue
            data = line.strip().split()
            x.append(float(data[0]))
            y.append(float(data[1]))
            z.append(float(data[2]))
        # for line in file:
        #     data = line.strip().split()
        #     x.append(float(data[0]))
        #     y.append(float(data[1]))
        #     z.append(float(data[2]))
    plt.plot(z)
    plt.xlabel('Index')
    plt.ylabel('Z Value')
    plt.title('Line Chart of Z Values')
    plt.show()

if __name__ == "__main__":
    # 指定文件夹路径
    # folder_path = 'data\\data\\3-22\\2nd'
    # 绘制压力数据
    # plot_pressure_data_allTime(folder_path)
    # 绘制伤口边缘数据
    # plot_wound_edge_data(folder_path)
    # 测试3-20数据量
    # plot_3_20_pressure_data(folder_path)
    # 3-22数据,测试查看全过程数据
    # plot_pressure_data(folder_path)
    # 绘制z轴数据
    plot_top_pos('data/data/status_train/topthe_top_pose.txt')