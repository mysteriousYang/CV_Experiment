# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchsummary import summary


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)  # default: stride=1, padding=0
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # default: stride=kernel_size
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)  
        self.fc1 = nn.Linear(16 * 2 * 2, 120)    # 注意特征图的尺寸
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        # print(x.shape)
        x = x.view(-1, 16 * 2 * 2)    # 展平操作，三维矩阵变为向量
        # print(x.shape)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
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

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
    x = np.array(cells)
    train_data = x.reshape(-1, 1, 20, 20).astype(np.float32)   
    k = np.arange(10)
    train_labels = np.repeat(k, 500)
    
    # 将numpy的ndarray转换为Torch的Tensor结构
    train_data = torch.tensor(train_data)
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    
    # 创建数据集对象
    dataset = TensorDataset(train_data, train_labels)
    
    # 按照7：3的比例划分训练集和验证集
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # 用DataLoader对象将训练集和测试集组织成不同批次（batch）
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 初始化模型、损失函数、优化器
    model = LeNet5()
    criterion = nn.CrossEntropyLoss()  # 定义分类损失函数为交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，将学习率设定为0.001
    
    # 训练阶段
    model.train()  # 设定模型为训练模式
    num_epochs = 1  # 总共训练多少轮
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            # print(inputs.shape)
            # print(labels.shape)
            outputs = model(inputs)             # [batch_size, in_channels, H, W]
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()                    
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
    
    # 测试阶段
    model.eval()  # 设定模型为测试模式（停止梯度更新）
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)   # 将one-hot向量转换为类别index
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 计算模型在测试集上的识别准确率
    accuracy = 100 * correct / total        
    print('模型分类的准确率为：{}%'.format(accuracy))

    print(summary(model, (1,20,20), device='cpu'))


if __name__ == '__main__':
    seed = 7
    torch.manual_seed(seed)   # 设置随机种子，使结果可复现
    main()