import os
import json
import numpy as np
from PIL import Image
import cv2
import random
import shutil

# 定义类别名称到标签值的映射
class_mapping = {
    "background": 0,
    "wound": 1
}

def labelme2mask(folder_path):
    # 指定包含JSON文件的文件夹路径
    # folder_path = "utils\\record_kinect_tools\\img"

    # 指定保存掩码图像和原始图像的目标文件夹路径
    # output_mask = "D:\\Program Files\\company\data\\lc_data\\origin\\lc_only_wound\\mask"
    # output_origin = "D:\\Program Files\\company\data\\lc_data\\origin\\lc_only_wound\\origin"
    # D:\Program Files\company\Jinjia\Projects\autoSuturePy\data\kinect\roi\1st\origin
    # output = folder_path
    output_mask = folder_path+ "_mask"
    output_origin = folder_path+ "_origin"
    os.makedirs(os.path.join(output_mask), exist_ok=True)
    os.makedirs(os.path.join(output_origin), exist_ok=True)
    print("文件创建完成")
    # 获取文件夹下所有JSON文件
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    for json_file in json_files:
        # 读取JSON文件
        with open(os.path.join(folder_path, json_file), "r") as f:
            data = json.load(f)

        # 获取图像尺寸
        width = data["imageWidth"]
        height = data["imageHeight"]

        # 创建单通道掩码图像
        mask = np.zeros((height, width), dtype=np.uint8)

        # 遍历JSON文件中的标注
        for annotation in data["shapes"]:
            label = annotation["label"]
            points = annotation["points"]
            polygon = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], class_mapping[label])

        # 保存单通道掩码图像到指定路径
        mask_image = Image.fromarray(mask)
        mask_filename = os.path.splitext(json_file)[0] + "_mask.png"
        mask_image.save(os.path.join(output_mask, mask_filename))

        # 保存原始图像到指定路径
        original_image_path = os.path.join(folder_path, data["imagePath"])
        original_image = Image.open(original_image_path)
        original_filename = os.path.splitext(json_file)[0] + "_original.jpg"
        original_image.save(os.path.join(output_origin, original_filename))

    print("transfor successfully")


def random_data(filePath):
    # 指定原始图像和标签的文件夹路径
    # filePath = "lc_data\\normal_distance\\"
    images_folder = filePath+"origin_origin"
    labels_folder = filePath+"origin_mask"

    # 计算划分的数据量
    image_files = os.listdir(images_folder)
    total_images = len(image_files)
    train_split = int(total_images * 0.8)
    test_split = int(total_images * 0.1)
    predict_split = total_images - train_split - test_split

    # 创建目标文件夹及子文件夹
    # 指定划分后的目标文件夹路径
    # makedir the data files
    train_folder = filePath+ "\\data\\train"
    os.makedirs(os.path.join(train_folder), exist_ok=True)
    test_folder = filePath+ "\\data\\test"
    os.makedirs(os.path.join(test_folder), exist_ok=True)
    predict_folder = filePath+ "\\data\\predict"
    os.makedirs(os.path.join(predict_folder), exist_ok=True)


    os.makedirs(os.path.join(train_folder, "image"), exist_ok=True)
    os.makedirs(os.path.join(train_folder, "label"), exist_ok=True)
    os.makedirs(os.path.join(test_folder, "image"), exist_ok=True)
    os.makedirs(os.path.join(test_folder, "label"), exist_ok=True)
    os.makedirs(os.path.join(predict_folder, "image"), exist_ok=True)
    os.makedirs(os.path.join(predict_folder, "label"), exist_ok=True)

    # 获取原始图像文件列表
    image_files = os.listdir(images_folder)

    # 随机打乱图像文件顺序
    random.shuffle(image_files)

    # 遍历图像文件并进行划分
    for i, image_file in enumerate(image_files):
        label_file = image_file.replace("_original.jpg", "_mask.png")
        
        if i < train_split:
            # 将图像文件和对应的标签文件拷贝到训练集文件夹
            shutil.copy(os.path.join(images_folder, image_file), os.path.join(train_folder, "image", image_file))
            shutil.copy(os.path.join(labels_folder, label_file), os.path.join(train_folder, "label", label_file))
        elif i < train_split + test_split:
            # 将图像文件和对应的标签文件拷贝到测试集文件夹
            shutil.copy(os.path.join(images_folder, image_file), os.path.join(test_folder, "image", image_file))
            shutil.copy(os.path.join(labels_folder, label_file), os.path.join(test_folder, "label", label_file))
        else:
            # 将图像文件和对应的标签文件拷贝到验证集文件夹
            shutil.copy(os.path.join(images_folder, image_file), os.path.join(predict_folder, "image", image_file))
            shutil.copy(os.path.join(labels_folder, label_file), os.path.join(predict_folder, "label", label_file))

    print("随机分配完成")

def rename_file(filePath):
    # 指定原始图像和标签的文件夹路径
    # filePath = "lc_data\\normal_distance\\"
    images_folder = filePath+"origin"
    labels_folder = filePath+"mask"

    # 获取原始图像文件列表
    image_files = os.listdir(images_folder)
    label_files = os.listdir(labels_folder)

    for i, image_file in enumerate(image_files):
        label_file = image_file.replace("_original.jpg", "_mask.png")
        os.rename(os.path.join(images_folder, image_file), os.path.join(images_folder, str(i)+".jpg"))
        os.rename(os.path.join(labels_folder, label_file), os.path.join(labels_folder, str(i)+".png"))
    print("rename successfully")


if __name__ == "__main__":
    # rename_file("data\\kinect\\roi\\1st\\")

    # 存放labelme的文件夹地方, 里面有json文件的同一级
    filepath = "data\\data\\status_train\\use_wound_roi\\origin"
    labelme2mask(filepath)
    outfile = "data\\data\\status_train\\use_wound_roi\\"
    random_data(outfile)

