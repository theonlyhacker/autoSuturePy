import os
import shutil
import json

# 取消对应txt文件名字同命名
def modifyMulti(filepath, mid):
    # 遍历文件夹中的所有文件
    file_count = 0
    for filename in os.listdir(filepath):
        # 检查文件是否以数字开头，以及是否是图片文件
        if filename.split('_')[0].isdigit() and filename.endswith('.jpg'):
            # 获取文件名中的数字部分
            file_num = int(filename.split('_')[0])
            # 检查文件名中的数字是否在指定的范围内
            if file_num < mid:
                new_status = 0
            else:
                new_status = 1
            # 构造新的文件名
            new_filename = f"{file_num}_roi_{new_status}.jpg"
            # 构造文件的完整路径
            # 这是图像路径
            old_imagepath = os.path.join(filepath, filename)
            new_imagepath = os.path.join(filepath, new_filename)
            # 这是txt路径
            new_txt_filename = f"{file_num}_pressure_{new_status}.txt"
            old_txt_name = f"{file_num}_pressure.txt"
            old_txtpath = os.path.join(filepath, old_txt_name)
            new_txtpath = os.path.join(filepath, new_txt_filename)
            # 重命名文件
            os.rename(old_imagepath, new_imagepath)
            os.rename(old_txtpath, new_txtpath)
            file_count += 1
    print("the number of files is : ", file_count)

def rename_and_move_data(source_dir, destination_dir):
    # 创建目标文件夹
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # 遍历源文件夹中的子文件夹
    for root, dirs, files in os.walk(source_dir):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            # 遍历子文件夹中的文件
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                # 如果是图像文件，将其移动并重命名
                if file.endswith('.jpg'):
                    new_filename = f"{subdir}_{file}"
                    new_file_path = os.path.join(destination_dir, new_filename)
                    shutil.copyfile(file_path, new_file_path)
                # 如果是文本文件，将其移动并重命名
                elif file.endswith('.txt'):
                    new_filename = f"{subdir}_{file}"
                    new_file_path = os.path.join(destination_dir, new_filename)
                    shutil.copyfile(file_path, new_file_path)


def modify(filepath, mid):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(filepath):
        # 检查文件是否以数字开头，以及是否是图片文件
        if filename.split('_')[0].isdigit() and filename.endswith('.jpg'):
            # 获取文件名中的数字部分
            file_num = int(filename.split('_')[0])
            # 检查文件名中的数字与 mid 的关系，设置对应的状态值
            if file_num < mid:
                new_status = 0
            else:
                new_status = 1
            # 构造新的文件名
            new_filename = f"{file_num}_roi_{new_status}.jpg"
            # 构造文件的完整路径
            old_imagepath = os.path.join(filepath, filename)
            new_imagepath = os.path.join(filepath, new_filename)
            # 重命名文件
            os.rename(old_imagepath, new_imagepath)
            print(f"Renamed {filename} to {new_filename}")

# 新增函数将所有图像输出到一个文件夹中，方便labelme打标签以及后续训练
# 用于训练roi的unet模型
def data2labelme(parent_folder):
    # 创建输出文件夹
    output_folder = os.path.join(parent_folder, "origin")
    os.makedirs(output_folder, exist_ok=True)
    count = 0
    # 遍历父文件夹下的所有子文件夹
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        # 如果是文件夹且名称包含 "collect_"
        if os.path.isdir(folder_path) and "collect_" in folder_name:
            # 遍历文件夹中的所有文件
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                # 如果是图片文件
                if os.path.isfile(file_path) and filename.lower().endswith(('.jpg')):
                    # 构造新的文件名
                    new_filename = f"{count}_roiColor.jpg".format(folder_name.replace("collect_", ""))
                    count += 1
                    # 复制并重命名图片文件到输出文件夹
                    shutil.copy(file_path, os.path.join(output_folder, new_filename))
                    # 检查是否存在对应的 JSON 文件
                    # json_filename = filename.replace("_roiColor.jpg", "_roiColor.json")
                    # json_file_path = os.path.join(folder_path, json_filename)
                    # if os.path.isfile(json_file_path):
                    #     # 构造新的 JSON 文件名
                    #     new_json_filename = new_filename.replace(".jpg", ".json")
                    #     # 复制并重命名 JSON 文件到输出文件夹
                    #     with open(json_file_path, 'r', encoding='utf-8') as json_file:
                    #         json_data = json.load(json_file)
                    #         with open(os.path.join(output_folder, new_json_filename), 'w', encoding='utf-8') as new_json_file:
                    #             json.dump(json_data, new_json_file)
                    # print("Moved and renamed:", filename, "->", new_filename)
    print(f"共有{count}张图像完成复制转移")


if __name__ == "__main__":
    #for unet 为了打标签的图像集合
    data2labelme("data\\points\\7-1\\6th\\1_copy")
    # Example usage:
    # 未缝合好标签记录为0，缝合好标签记录为1
    # modify("data\\data\\status_train\\4-26-img\\collect_7", 45)
    # modifyMulti("data\\data\\status_train\\4-26-img\\collect_7", 110)
    # 调用函数
    # filepath = "data\\data\\status_train\\multi"
    # rename_and_move_data(filepath+"\\1st", filepath+"\\out")
    pass