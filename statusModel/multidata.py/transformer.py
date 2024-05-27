import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# 定义数据加载器
class MyDataset(Dataset):
    def __init__(self, img_txt_dir, transform=None):
        self.img_txt_dir = img_txt_dir
        # self.img_names = os.listdir(img_txt_dir)
        self.img_names = [f for f in os.listdir(img_txt_dir) if f.endswith(('.jpg'))]
        self.transform = transform
        self.last_values = [0, 0, 0]  # 用于保存上一次读取到的数值

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_txt_name = self.img_names[idx]
        img_path = os.path.join(self.img_txt_dir, img_txt_name)
        text_file = f"{img_txt_name.split('_')[0]}_{img_txt_name.split('_')[1]}_pressure_{img_txt_name.split('_')[-1]}".replace('.jpg', '.txt')
        text_path = os.path.join(self.img_txt_dir, text_file)
        label = int(img_txt_name.split('_')[-1].split('.')[0]) # 提取标签
        # 读取文档数据
        values = []
        with open(text_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:  # 如果不为空
                    values.append(float(line))
        # 如果数据中没有有效值，则跳过该样本
        if not values:
            return self.__getitem__((idx + 1) % len(self))  # 递归调用 __getitem__ 方法，跳过当前样本
        # 读取图片
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # 复制填充
        while len(values) < 3:
            values += values
        # 截断或补充到3个数值
        values = values[:3]
        # 将数据转换为Tensor
        lines = torch.tensor(values, dtype=torch.float32)
        return img, lines, label

class MultimodalModel(nn.Module):
    def __init__(self, num_classes, cnn_hidden_size=512, text_input_size=3):
        super(MultimodalModel, self).__init__()
        # 图像数据处理模块（CNN）
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # 移除最后一层全连接层
        self.cnn_hidden_size = cnn_hidden_size
        # 文本数据处理模块（全连接层）
        self.text_fc = nn.Linear(text_input_size, 128)  # 假设每个文本数据有3个数值
        self.text_hidden_size = 128
        # 融合模块
        self.fc_fusion = nn.Linear(cnn_hidden_size + self.text_hidden_size, num_classes)
        
    def forward(self, images, texts):
        # 图像数据处理
        cnn_features = self.cnn(images)
        
        # 文本数据处理
        text_features = self.text_fc(texts)
        
        # 特征融合
        combined_features = torch.cat((cnn_features, text_features), dim=1)
        
        # 最终分类或回归
        output = self.fc_fusion(combined_features)
        return output


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    best_loss = float('inf')
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, texts, labels in train_loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, texts)
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
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "pth\\status\\transformer&cnn.pth")
            print("Model saved!")
    print('Training complete!')


# 定义训练参数
img_txt_dir = 'data\\data\\status_train\\multi\\out'
batch_size = 4
epochs = 10

# 创建数据集和数据加载器
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = MyDataset(img_txt_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型和损失函数
model = MultimodalModel(num_classes=2)  # 假设有2个类别
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数，因为标签是分类问题
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_model(model, dataloader, criterion, optimizer, epochs)