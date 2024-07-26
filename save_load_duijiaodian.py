#保存两个对角点到文件
def save_points_to_file(points, file_path):
          with open(file_path, 'w') as f:
              for point in points:
                  f.write(f"{point[0]},{point[1]}\n")
#读取对角点文件到代码中
def load_points_from_file(file_path):
    load_points = []
    with open(file_path, 'r') as f:
         lines = f.readlines()
         for line in lines :
             a = line.split(",")
             load_points.append(int(a[0]))
             load_points.append(int(a[1]))
    width = load_points[2]-load_points[0]
    height = load_points[3]-load_points[1]
    x,y = load_points[0],load_points[1]
    return x,y,width,height