# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


if __name__ == '__main__':
    # 获取代码文件的绝对路径
    file_dir = os.path.dirname(os.path.abspath(__file__))

    # 设置当前工作目录为代码文件所在的目录
    os.chdir(file_dir)

    # digit
    # # 检查文件是否存在
    # if not os.path.isfile('digits.png'):
    #     print("File not found")
    
    # image = cv.imread('digits.png')    # 5000个手写数字的图像，图像大小为2000×1000，每个手写数字的尺寸为20×20.
    # # 判断是否成功读取图像
    # if image is None:
    #     print('Failed to read digits.png.')
    #     sys.exit()

    # olivetti face
    # 检查文件是否存在
    if not os.path.isfile('olivettifaces.png'):
        print("File not found")
    
    image = cv.imread('olivettifaces.png')
    # 判断是否成功读取图像
    if image is None:
        print('Failed to read olivettifaces.png.')
        sys.exit()

    # 转为灰度图像
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 读取图像中的数据并创建训练数据

    # digit
    # cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]  # 将图像切分为50行、100列，得到每幅子图像

    # face
    cells = [np.hsplit(row, 20) for row in np.vsplit(gray, 20)]

    x = np.array(cells)
    # 创建训练数据
    train_data = x.reshape(-1, 57*47).astype(np.float32)
    #train_data = x.reshape(-1, 20*20).astype(np.float32)
    # 创建训练标签

    # digit
    # k = np.arange(10)  # 共10类（0-9个数字）
    # train_labels = np.repeat(k, 500)   # 每个数字500个样本

    # face
    k = np.arange(40) 
    train_labels = np.repeat(k, 10)  

    # 划分数据为训练集和测试集（7:3比例）
    train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    
    model = GaussianNB()
    model.fit(train_data,train_labels)

    labels_pred = model.predict(test_data)

    print(classification_report(labels_pred,test_labels))