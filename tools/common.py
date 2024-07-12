from tools.calibration import cal_trans_data


def com_cal_trans_data(data):
    print("into common tran_data 接口")
    result = cal_trans_data(data)
    return result

def plot_plan_points():
    print("into common plot_rm65_points 接口")
    # 从两个txt文件从进行读取然后绘制--2d层面
    