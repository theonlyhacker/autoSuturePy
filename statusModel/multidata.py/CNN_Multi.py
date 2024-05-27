import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

import cv2
# 获取图像大小和形状
def get_image_shape(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    # 查找第一个图像文件
    for file in files:
        if file.endswith(('.jpg')):
            image_path = os.path.join(folder_path, file)
            # 读取图像
            image = cv2.imread(image_path)
            if image is not None:
                height, width, channels = image.shape
                # 返回图像形状,pytorch主要返回CHW的格式。
                return channels, height, width
    

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, folder_path, max_num_sensors, transform=None):
        self.folder_path = folder_path
        self.max_num_sensors = max_num_sensors
        self.transform = transform
        self.data = self.load_data()
        self.input_shape = get_image_shape(folder_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, pressure, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, pressure, label

    def load_data(self):
        data = []
        max_pressure_length = 3  # 设置压力传感器数据的最大长度
        previous_pressure = None
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith(".jpg"):
                image_path = os.path.join(self.folder_path, file_name)
                image = cv2.imread(image_path)
                # 转换图像
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                data_number = file_name.split("_")[0]
                pressure_file_name = f"{data_number}_pressure_{file_name[-5]}.txt"
                pressure_path = os.path.join(self.folder_path, pressure_file_name)

                # 读取压力传感器数据
                if os.path.exists(pressure_path):
                    with open(pressure_path, "r") as file:
                        pressure_data = file.readlines()
                        pressure_data = [float(p.strip()) for p in pressure_data]
                else:
                    pressure_data = previous_pressure if previous_pressure is not None else []
                # 填充或截断压力传感器数据，使其长度一致
                pressure_data = pressure_data[:max_pressure_length]  # 截断数据
                pressure_data += [0] * (max_pressure_length - len(pressure_data))  # 填充数据
                # 将列表转换为张量
                pressure_data = torch.tensor(pressure_data, dtype=torch.float32)

                data.append((image, pressure_data, int(file_name[-5])))
                previous_pressure = pressure_data

        return data

# 定义模型类
class CustomModel(nn.Module):
    def __init__(self, input_shape, max_num_sensors):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # 这里fc1的输入特征数量需要修改为 167552，也就是 32*62*62？
        self.fc1 = nn.Linear(167552 + max_num_sensors, 64)  # 修改输入特征数量为 167555
        self.fc2 = nn.Linear(64, 1)


    def forward(self, x_image, x_pressure):
        x_image = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x_image), 2))
        batch_size = x_image.size(0)
         # 将特征图展平--未展平之前是torch.Size([2, 32, 44, 119])，也就是32*44*119的大小，好像是卷积的特征图大小？
        x_image = x_image.view(batch_size, -1) 
        # print("the image_shape: ",x_image.shape)  # 检查张量的形状
        # print("the pressure_shape: ",x_pressure.shape)  # 检查张量的形状
        # x_combined = torch.cat((x_image, x_pressure), dim=1)  # 组合图像数据和压力传感器数据
        
        # 将 x_image 调整为与 x_pressure 相同的形状
        # x_image = x_image.view(batch_size, 1, -1)  # 将特征图展平并增加一个维度
        # # 组合图像数据和压力传感器数据
        # x_combined = torch.cat((x_image, x_pressure.unsqueeze(1)), dim=2)  # 在维度1上进行拼接
        
        # 将 x_image 调整为与 x_pressure 相同的形状

        # 组合图像数据和压力传感器数据
        x_combined = torch.cat((x_image, x_pressure), dim=1)  # 在维度1上进行拼接
        # print("the combined_shape: ",x_combined.shape)  # 检查张量的形状

        x_combined = nn.functional.relu(self.fc1(x_combined))
        x_combined = torch.sigmoid(self.fc2(x_combined))
        return x_combined



# 设置随机种子
torch.manual_seed(42)

# 文件夹路径
folder_path = "data\\data\\status_train\\1st\\output"

# 压力传感器最大数量
max_num_sensors = 3  # 假设最大数量为3

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = CustomDataset(folder_path, max_num_sensors, transform=transform)

# 定义数据集大小
dataset_size = len(dataset)

# 定义划分比例
train_ratio = 0.8
test_ratio = 1 - train_ratio

# 计算划分大小
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

# 划分数据集
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

# 划分训练集和测试集
# train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# 创建数据加载器
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
test_loader = DataLoader(test_set, batch_size=2, shuffle=False)

# 创建模型
model = CustomModel(dataset.input_shape, max_num_sensors)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置TensorBoard
writer = SummaryWriter()

# 训练模型
num_epochs = 20
# best_loss统计，初始化为正无穷
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, pressures, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images, pressures)
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    writer.add_scalar('training_loss', epoch_loss, epoch)    
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'statusModel\\best_model.pth')


# 保存模型权重
# torch.save(model.state_dict(), 'statusModel\\model_weights.pth')
