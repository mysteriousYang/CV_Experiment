# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':
    # 获取代码文件的绝对路径
    file_dir = os.path.dirname(os.path.abspath(__file__))

    # 设置当前工作目录为代码文件所在的目录
    os.chdir(file_dir)

    # 检查文件是否存在
    if not os.path.isfile('digits.png'):
        print("File not found")
    
    image = cv.imread('digits.png')    # 5000个手写数字的图像，图像大小为2000×1000，每个手写数字的尺寸为20×20.
    # 判断是否成功读取图像
    if image is None:
        print('Failed to read digits.png.')
        sys.exit()

    # 转为灰度图像
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 读取图像中的数据并创建训练数据
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]  # 将图像切分为50行、100列，得到每幅子图像
    x = np.array(cells)
    # 创建训练数据
    train_data = x.reshape(-1, 400).astype(np.float32)
    # 创建训练标签
    k = np.arange(10)  # 共10类（0-9个数字）
    train_labels = np.repeat(k, 500)   # 每个数字500个样本

    # 划分数据为训练集和测试集（7:3比例）
    train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.3, random_state=42)
    
    # 利用sklearn建立KNN分类器
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_data, train_labels)

    # 预测测试集
    predicted_labels = knn.predict(test_data)

    # 计算模型在测试集上的识别准确率
    accuracy = np.mean(predicted_labels == test_labels) * 100
    print('模型分类的准确率为：{}%'.format(accuracy))

    test_img = cv.imread("digit_test.png", cv.IMREAD_GRAYSCALE)

    print("模型的预测结果为: ", knn.predict(test_img.reshape(-1, 400).astype(np.float32)))
    cv.namedWindow("Digit 5", cv.WINDOW_NORMAL | cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow("Digit 5", 400, 400)
    cv.imshow("Digit 5", test_img)
    cv.waitKey()