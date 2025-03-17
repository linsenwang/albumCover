import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from PIL import Image, ImageDraw

# 计算 Bézier 曲线
def bezier_curve(control_points, n_points=100):
    n = len(control_points) - 1
    t = np.linspace(0, 1, n_points)
    curve = np.zeros((n_points, 2))
    
    for i, P in enumerate(control_points):
        curve += np.outer(comb(n, i) * (t**i) * ((1 - t)**(n - i)), P)
    
    return curve

def round_corner(image):

    w, h = (image.size[0] * 10, image.size[1] * 10)

    # r =300  # 增大r，以适应更大的像素分辨率
    r = min(w, h) / 20  # 圆角半径
    s = 0.3
    # h = 6000
    # w = 6000
    img_size = (6000, 6000)  # 生成的遮罩图像大小

    control_points1 = np.array([[-r, 0], [-r, r * s], [-r, r], [- r * s, r], [0, r]]) + np.array([- w / 2, h / 2])#左上角
    control_points2 = np.array([[0, r], [r * s, r], [r, r], [r, r * s], [r, 0]]) + np.array([w / 2, h / 2]) #右上角
    control_points3 = np.array([[r, 0], [r, -r * s], [r, -r], [r * s, -r], [0, -r]]) + np.array([w / 2, - h / 2]) #右下角
    control_points4 = np.array([[0, -r], [-r * s, -r], [-r, -r], [-r, -r * s], [-r, 0]]) + np.array([- w / 2, - h / 2]) #左下角

    # edge1 = np.array([[- w / 2, h / 2], [w / 2, h / 2]]) + np.array([0, r]) #上边
    # edge2 = np.array([[- w / 2, - h / 2], [w / 2, - h / 2]]) + np.array([0, -r]) #下边
    # edge3 = np.array([[- w / 2, h / 2], [- w / 2, - h / 2]]) + np.array([- r, 0]) #左边
    # edge4 = np.array([[w / 2, h / 2], [w / 2, - h / 2]]) + np.array([r, 0]) #右边

    edge1 = np.array([[- w / 2, h / 2], [w / 2, h / 2]]) + np.array([0, r]) # 上边
    edge2 = np.array([[w / 2, h / 2], [w / 2, - h / 2]]) + np.array([r, 0]) # 右边
    edge3 = np.array([[w / 2, - h / 2], [- w / 2, - h / 2]]) + np.array([0, -r]) # 下边
    edge4 = np.array([[- w / 2, - h / 2], [- w / 2, h / 2]]) + np.array([- r, 0]) # 左边

    # 生成曲线
    curve1 = bezier_curve(control_points1)
    curve2 = bezier_curve(control_points2)
    curve3 = bezier_curve(control_points3)
    curve4 = bezier_curve(control_points4)

    edge1 = bezier_curve(edge1)
    edge2 = bezier_curve(edge2)
    edge3 = bezier_curve(edge3)
    edge4 = bezier_curve(edge4)

    # # 绘图
    # # fig, ax = plt.subplots(figsize=(60, 60))
    # fig, ax = plt.subplots(figsize=(16, 16))
    # plt.plot(curve1[:, 0], curve1[:, 1], 'y-')
    # plt.plot(curve2[:, 0], curve2[:, 1], 'y-')
    # plt.plot(curve3[:, 0], curve3[:, 1], 'y-')
    # plt.plot(curve4[:, 0], curve4[:, 1], 'y-')

    # plt.plot(edge1[:, 0], edge1[:, 1], 'y-')
    # plt.plot(edge2[:, 0], edge2[:, 1], 'y-')
    # plt.plot(edge3[:, 0], edge3[:, 1], 'y-')
    # plt.plot(edge4[:, 0], edge4[:, 1], 'y-')

    # ax.set_aspect('equal')
    # plt.show()

    # **按顺序拼接点**
    ordered_points = np.vstack([
        curve1,  # 左上角圆角
        edge1,   # 上边
        curve2,  # 右上角圆角
        edge2,   # 右边
        curve3,  # 右下角圆角
        edge3,   # 下边
        curve4,  # 左下角圆角
        edge4    # 左边
    ])

    # **转换到 PIL 的像素坐标**
    min_x, min_y = ordered_points.min(axis=0)
    max_x, max_y = ordered_points.max(axis=0)

    def transform_to_pixel(point):
        """将数学坐标转换为像素坐标"""
        x = int((point[0] - min_x) / (max_x - min_x) * (img_size[0] - 1))
        y = int((1 - (point[1] - min_y) / (max_y - min_y)) * (img_size[1] - 1))  # 反转y坐标
        return (x, y)

    # 转换所有 Bézier 轮廓点
    pixel_points = [transform_to_pixel(p) for p in ordered_points]

    # **创建 PIL 遮罩**
    mask = Image.new("L", img_size, 0)  # 创建黑色背景的遮罩
    draw = ImageDraw.Draw(mask)

    # **绘制完整封闭形状**
    draw.polygon(pixel_points, fill=255)

    # 显示遮罩
    # mask.show()
    resample_ratio = 4
    resample_size = (image.size[0] * resample_ratio, image.size[1] * resample_ratio)
    mask = mask.resize(resample_size, Image.LANCZOS)  # 调整遮罩大小
    inter = image.resize(resample_size, Image.LANCZOS)  # 调整遮罩大小
    inter.putalpha(mask)  # 设置透明度
    image = inter.resize(image.size, Image.LANCZOS)  # 调整遮罩大小
    return image
    # image.show()
    # image.save("./continuous_rounded_corner.png")






