import os
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
            new_filename = f"{file_num}_roi_{new_status}_1212.jpg"
            # 构造文件的完整路径
            old_imagepath = os.path.join(filepath, filename)
            new_imagepath = os.path.join(filepath, new_filename)
            # 重命名文件
            os.rename(old_imagepath, new_imagepath)
            print(f"Renamed {filename} to {new_filename}")


modify('C://Users//legion//Desktop//data1//test//image',-1)