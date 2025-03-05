import numpy as np
import cv2 as cv
import os
import sys

def digit():
        # 获取代码文件的绝对路径
    file_dir = os.path.dirname(os.path.abspath(__file__))

    # 设置当前工作目录为代码文件所在的目录
    os.chdir(file_dir)

    digit
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

    # digit
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]  # 将图像切分为50行、100列，得到每幅子图像
    x = np.array(cells)

    data = x.reshape(-1, 20*20).astype(np.float32)
    # 创建训练标签

    digit
    k = np.arange(10)  # 共10类（0-9个数字）
    labels = np.repeat(k, 500)   # 每个数字500个样本

    return data, labels

def olivetti_face():
    # 获取代码文件的绝对路径
    file_dir = os.path.dirname(os.path.abspath(__file__))

    # 设置当前工作目录为代码文件所在的目录
    os.chdir(file_dir)

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

    # face
    cells = [np.hsplit(row, 20) for row in np.vsplit(gray, 20)]

    x = np.array(cells)
    # 创建训练数据
    data = x.reshape(-1, 57*47).astype(np.float32)
    # 创建训练标签

    # face
    k = np.arange(40) 
    labels = np.repeat(k, 10)  

    return data,labels