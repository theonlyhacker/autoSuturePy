import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.jpg'))
        # print("\n-------self.imgs_path-----------\n",self.imgs_path)

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # print("-----------origin path-------------\n",image_path)
        # print('\n')
        # 根据image_path生成label_path
        # label_path = image_path.replace('image\*_original.jpg', 'label\*_mask.png')
        label_path = image_path.replace("\\image\\", "\\label\\").replace("_original.jpg", "_mask.png")
        # print(label_path)
        # print("--------image or label---------")
        # print(label_path)
        # 读取训练图片和标签图片

        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # get roi 保存的时候只保存了roi区域，因此不需要这里再筛选 --modify by liuchao 3-7
        # x,y,w,h= 1020,500,300,130
        # image = image[y:y+h, x:x+w]
        # label = label[y:y+h, x:x+w]

        # # 将数据转为单通道的图片
        # 原图是BGR还是RGB？
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("image",image)
        # cv2.imshow("label",label)
        # cv2.waitKey(0)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # print(label.shape)

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

    
if __name__ == "__main__":
    # data_path = "data/train/"
    # image_path = os.path.join(data_path, 'image/*.png')
    # imgs_path = glob.glob(r'../data/train/image/*.png')
    # print('length of image', len(imgs_path))


    isbi_dataset = ISBI_Loader(r"D:\Program Files\company\Jinjia\Projects\autoSuturePy\Unet\data\3_7\data\test")
    print("数据个数：", len(isbi_dataset))
    isbi_dataset[0]
    # train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
    #                                            batch_size=4,
    #                                            shuffle=False)
    # for image, label in train_loader:
    #     print(image.shape)
    #     pass

