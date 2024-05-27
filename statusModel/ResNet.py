import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from customData import CustomDataset  # 导入自定义的数据集类
from PIL import Image
import cv2

def train(model, train_loader, criterion, optimizer, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    best_loss = float('inf')
    best_model_path = 'pth\\status\\resnet50_model_test.pth'
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 清除梯度
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)
            print("Model saved!")
    return model

def predict(image):
    # 加载预训练的 ResNet50 模型
    model = models.resnet50(pretrained=False)  # 不加载预训练权重
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 输出类别数为2
    
    # 加载之前训练好的模型权重
    model.load_state_dict(torch.load('pth\\status\\resnet50_model.pth'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
        # print(f'Predicted class: {predicted_class}')
    return predicted_class

if __name__ == "__main__":
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # train---------------------------------------------------
    # 加载数据集
    # train_dataset = CustomDataset(root_dir='data\\data\\status_train\\2nd_copy')
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # # 加载预训练的 ResNet50 模型
    # model = models.resnet50(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = True
    # # 修改全连接层，使其适应当前问题
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)  # 输出类别数为2
    # # 定义优化器和损失函数
    # optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()
    # # 训练模型
    # trained_model = train(model, train_loader, criterion, optimizer, num_epochs=30)
    
    
    # 测试预测
    # datapath = "data\\data\\status_train\\1st\\collect_6\\"
    # image1_path = datapath + '135_roi_1.jpg'
    image1_path = 'data\\data\\3-19\\2nd\\collect_1\\20\\roi_color.jpg'
    # image = Image.open(image1_path)  # 替换为测试图片的路径
    # 使用 OpenCV 读取图像
    image = cv2.imread(image1_path)
    # OpenCV 读取的图像为 BGR 格式，需要转换为 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 将图像转换为 PIL 图像
    image = Image.fromarray(image)
    image = transform(image)
    predicted_class = predict(image)
    print(f'Predicted class: {predicted_class}')
