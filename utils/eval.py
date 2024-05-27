import glob
import numpy as np
import torch
import torch.nn as nn
import os
import cv2
import math
from model.unet_model import UNet
from model.arch import UNext
from utils.dataset import ISBI_Loader
from model.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork



def Get_Dice(predict, target):
    # predict = torch.where(predict > 0, 0, 1)
    # target = torch.where(target > 0, 0, 1)
    predict = predict.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    inner = np.sum((predict * target) == 1)
    union = np.sum(predict == 1) + np.sum(target == 1)
    dice_value = 2 * inner / union
    iou_value = (inner + 0.00001) / (union-inner + 0.00001)
    return iou_value, dice_value

# 选择设备，有cuda用cuda，没有就用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载网络，图片单通道，分类为1。
#net = UNet(n_channels=1, n_classes=1)
net = UNext(num_classes=1)
# 将网络拷贝到deivce中
net.to(device=device)
# 加载模型参数
net.load_state_dict(torch.load('best_modelunext.pth', map_location=device))
#net.load_state_dict(torch.load('best_modelunext.pth', map_location=device))
# 测试模式
net.eval()

isbi_dataset = ISBI_Loader("./data/test/")
print("数据个数：", len(isbi_dataset))
train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                           batch_size=1,
                                           shuffle=False)
#Unet的训练
Dicelist = []
Ioulist = []
for image, label in train_loader:
    image = image.to(device="cuda:0",dtype=torch.float32)
    label = label.to(device="cuda:0",dtype=torch.float32)
    pred = net(image)
    sigmoid = nn.Sigmoid()
    pred = sigmoid(pred)
    pred = torch.where(pred>0.5,1,0)
    # pred = np.array(pred.data.cpu()[0],dtype=np.float32)[0]
    # print(pred.shape)
    # cv2.imshow("image", pred)
    # cv2.waitKey(0)

    Iou,Dice = Get_Dice(pred,label)
    Iou = Iou.tolist()
    Dice = Dice.tolist()
    Dicelist.append(Dice)
    Ioulist.append(Iou)
print(f"Dice = {sum(Dicelist)/len(Dicelist)}%",f"Iou = {sum(Ioulist)/len(Ioulist)}%")