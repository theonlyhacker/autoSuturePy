import os
import torchvision.transforms as transforms
from torch.utils.data import  Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToTensor()  # 转换为张量
        ])
        self.image_folders = [os.path.join(root_dir, folder) for folder in sorted(os.listdir(root_dir))]
        self.images = []
        self.labels = []
        # 读取第一张图像获取形状
        # test_file_path = os.path.join(self.image_folders[0], sorted(os.listdir(self.image_folders[0]))[0])
        # first_image = Image.open(os.path.join(self.image_folders[0], sorted(os.listdir(self.image_folders[0]))[0]))
        first_image = self.find_first_jpg_image()
        self.image_shape = first_image.size[::-1]  # 图像大小 (width, height)
        # 获取图像通道数
        num_channels = len(first_image.getbands())
        # 将通道数添加到图像形状中
        self.image_shape = (num_channels,) + self.image_shape

        first_image.close()
        for folder in self.image_folders:
            images = [img for img in sorted(os.listdir(folder)) if img.endswith('.jpg')]
            for i, img in enumerate(images):
                image_path = os.path.join(folder, img)
                self.images.append(image_path)
                self.labels.append(int(img[-5]))  # 根据文件名后缀确定标签（0或1）

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        init_image_path = self.images[idx]
        init_image = Image.open(init_image_path)
        init_image = self.transform(init_image)
        label = self.labels[idx]
        return init_image, label, self.image_shape
    
    def find_first_jpg_image(self):
        for folder in self.image_folders:
            files_in_folder = sorted(os.listdir(folder))
            jpg_files = [file for file in files_in_folder if file.lower().endswith('.jpg')]
            if jpg_files:
                first_jpg_file = jpg_files[0]
                first_image_path = os.path.join(folder, first_jpg_file)
                return Image.open(first_image_path)