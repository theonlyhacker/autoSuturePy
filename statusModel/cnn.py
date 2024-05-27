import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from statusModel.customData import CustomDataset
from PIL import Image
import cv2
import time


class CustomCNN(nn.Module):
    def __init__(self, input_shape):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 计算全连接层的输入维度
        conv_output_size = self._get_conv_output_size(input_shape)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def _get_conv_output_size(self, shape):
        x = torch.rand(1, *shape)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_loss = float('inf')
    best_acc = 0
    best_model_path = 'pth\\status\\status_cnn.pth'
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs1, labels, image_shape in dataloader:
            inputs1, labels = inputs1.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs1.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        # 保存最佳模型
        if epoch_loss < best_loss and epoch_acc > best_acc:
            best_loss = epoch_loss
            best_acc = epoch_acc
            torch.save(model.state_dict(), best_model_path)
            print("Model saved!")


# 定义预测函数
def predict_status(model, image):
    with torch.no_grad():
        image1 = image.unsqueeze(0)# 增加 batch 维度
        outputs = model(image1)# 将图像数据传递给模型
        # 获取预测结果
        _, predicted = torch.max(outputs, 1)
        return predicted.item()


def train_cnn():
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #  定义数据集和数据加载器
    root_dir = "data\\data\\status_train\\4-26-img"
    dataset = CustomDataset(root_dir)
    input_shape = dataset.image_shape
    # print(f"the shape of input is :{input_shape}")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # 初始化模型、损失函数和优化器
    model = CustomCNN(input_shape).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练模型
    train_model(model, dataloader, criterion, optimizer)

def pre_cnn():
    # 传入需要预测的图像路径 
    image1_path = 'data\\data\\status_train\\use_wound_roi\\collect_2\\144_roi_0.jpg'
    # image1_path = datapath + '280_roi_1.jpg'
    image = cv2.imread(image1_path)
    # OpenCV 读取的图像为 BGR 格式，需要转换为 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 将图像转换为 PIL 图像
    image = Image.fromarray(image)
    input_shape = (len(image.getbands()),)+image.size[::-1]
    # print("----",input_shape)
    # 加载预训练模型
    model = CustomCNN(input_shape)
    model.load_state_dict(torch.load('pth\\status\\status_cnn.pth'))
    model.eval()
    # 图像预处理转换
    transform = transforms.Compose([
        transforms.ToTensor()  # 转换为张量
    ])
    image1 = transform(image)
    predicted_status = predict_status(model,image1)


if __name__ == "__main__":
    # 训练
    # train_cnn()
    # 预测
    # 预测一千次看所用时间
    start_time = time.time()
    for i in  range(100):
        pre_cnn()
        print(f"第{i}次")
    end_time = time.time()
    print(f"当前所用时间是{end_time-start_time}")
    
