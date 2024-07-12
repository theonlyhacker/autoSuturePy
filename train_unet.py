from model.unet_model import UNet
from utils.dataset import ISBI_Loader
from utils.loss import DiceLoss
from torch import optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from model.arch import UNext,UNext_L
from torch.utils.data import random_split
from model.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os


#原来的epoch是40 batch_size=1
def train_net(net, device, train_data_path,test_data_path, epochs=20, batch_size=8, lr=0.0001):
    # 加载训练集
    train_isbi_dataset = ISBI_Loader(train_data_path)
    test_isbi_dataset = ISBI_Loader(test_data_path)
    #train_dataset, test_dataset = random_split(dataset=isbi_dataset, lengths=[16, 4],generator=torch.Generator().manual_seed(0))

    train_loader = torch.utils.data.DataLoader(dataset=train_isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_isbi_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    # writer=SummaryWriter(log_dir=r'Unet/runs',comment=f'Bce_{epochs}')
    writer=SummaryWriter(comment=f'Bce_{epochs}')
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = DiceLoss()


    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    global_step=0
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for image, label in train_loader:
            global_step+=1
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # pred = torch.cat((pred, image), dim=1)
            # pred = net1(pred)
            # 计算loss
            # loss = 0.5 * criterion1(pred, label) +criterion2(pred,label)

            loss = criterion1(pred, label)
            print('Loss/train', loss.item())
            writer.add_scalar("Loss/train",loss.item(),global_step)
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                #只保存模型参数
                CUR_PATH=os.path.dirname(os.path.realpath(__file__))
                save_pth_path = os.path.join(CUR_PATH, "pth\\roi\\unet.pth")
                torch.save(net.state_dict(), save_pth_path)
                #保存模型和模型参数
                #torch.save(net,'best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()

        test_loss_list = []
        for image, label in test_loader:
            with torch.no_grad():
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                # 使用网络参数，输出预测结果
                pred = net(image)
                # 计算loss
                # test_loss = 0.5 * criterion1(pred, label) +criterion2(pred,label)

                test_loss = criterion1(pred, label)
                test_loss_list.append(test_loss)

        mean_test_loss = sum(test_loss_list) / len(test_loss_list)
        writer.add_scalar("Loss/test", mean_test_loss.item(), global_step)
        print("epoch: ", epoch + 1, "  global_step: ", global_step, '   Loss/test', mean_test_loss.item())


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    # net = UNext(num_classes=1)
    net = UNet(n_classes=1,n_channels=1)
    # net = ReconstructiveSubNetwork(in_channels=1, out_channels=1)
    # net1 = DiscriminativeSubNetwork(in_channels=2, out_channels=1)
    #net = UNext_L(num_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # net1.to(device=device)
    # 指定训练集地址，开始训练
    CUR_PATH=os.path.dirname(os.path.realpath(__file__))
    # print("CUR_PATH",CUR_PATH)
    # exit(0)
    filepath = "data\\points\\7-1\\6th\\1_copy\\data"
    train_data_path = os.path.join(CUR_PATH, filepath+"\\train\\")
    test_data_path = os.path.join(CUR_PATH, filepath+"\\test\\")
    train_net(net, device, train_data_path=train_data_path,test_data_path=test_data_path)