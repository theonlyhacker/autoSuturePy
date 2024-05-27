from getKinect import KinectCapture
import cv2
import time
import numpy as np

def get_colorImg():
     kinect = KinectCapture()
     time.sleep(4)
     color, depth = kinect.get_frames()
     cv2.imwrite('data\\kinect\\roi\\test\\color.jpg', color)
    #  cv2.imshow('depth', depth)
    #  cv2.waitKey(0)
    #  cv2.imwrite('origin_depth.png', depth)
    #  cv2.imwrite('origin_color.png', color)

def getROI(img):
     # 选取ROI区域
     r = cv2.selectROI(img)
     print(r)
     # 选取的区域
     roi = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
     cv2.imshow('roi', roi)
     cv2.waitKey(0)
     cv2.imwrite('roi.png', roi)

def proImg(x,y,w,h,origin_img, filePath):
    # 提取ROI区域
    roi = origin_img[y:y+h, x:x+w].copy()
    # 转成灰度图 
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blurred_roi = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    # 利用Canny抽取边缘
    roi_edges = cv2.Canny(blurred_roi, 45, 70, 3)
    # 执行形态学操作以连接断开的边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 动态修改腐蚀膨胀次数完成边缘抽取
    for i in range(5):
        roi_edges = cv2.dilate(roi_edges, kernel)  # 膨胀
    for i in range(1):
        roi_edges = cv2.erode(roi_edges, kernel)  # 腐蚀
    # 执行连通组件分析
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_edges)
    # 找到面积最大的连通域
    max_area = 0
    max_area_label = -1
    for i in range(1, len(stats)):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_area_label = i

    output_img = origin_img.copy()
    # 提取最大连通域的边缘点
    contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 保存最大轮廓区域内的像素坐标
    maxContourPoints = []
    if len(contours) > 0:
        maxContourPoints = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]
    # 创建一个与输入图像大小相同的二值图像，用于将最大轮廓内的像素点标记为白色
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    if len(maxContourPoints) > 0:
        cv2.drawContours(mask, [maxContourPoints], -1, 255, cv2.FILLED)

    small_label = roi.copy()
    # 提取最大轮廓内的像素点
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 255:
                origin_x = j + x
                origin_y = i + y
                pt = (origin_x, origin_y)  # 修正坐标顺序
                output_img = cv2.circle(output_img, pt, 1, (255, 255, 255), -1)  # 使用圆形标记像素点为白色
               #  small_label = cv2.circle(small_label, (j , i), 1, (255, 255, 255), -1)
            else:
                output_img = cv2.circle(output_img, (j + x,i + y), 1, (0, 0, 0), -1)
               #  small_label = cv2.circle(small_label, (j , i), 1, (0, 0, 0), -1)

    # cv2.imshow("Image with Contours", output_img)
    # 绘制最大连通域的边缘轮廓
    imageWithContours = roi.copy()
    contourColor = (0, 255, 0)  # 设置轮廓颜色为绿色，可根据需要进行调整
    contourThickness = 1  # 设置轮廓线宽，可根据需要进行调整
    cv2.drawContours(imageWithContours, [maxContourPoints], -1, contourColor, contourThickness)
    # 显示带有轮廓的图像
    cv2.imshow("Image with Contours", imageWithContours)
    cv2.waitKey(0)
    # 其他地方定位黑色
    output_img[:y, :] = 0
    output_img[y+h:, :] = 0
    output_img[:, :x] = 0
    output_img[:, x+w:] = 0
    cv2.imwrite(filePath, output_img)




if __name__ == "__main__":
    get_colorImg()
    exit(0)
    # getROI(cv2.imread('origin_color.png'))
    proImg(1014,324,340,180,cv2.imread('origin_color.png'), 'D:\\Program Files\\company\\Jinjia\\Projects\\autoSuture\\test.png')