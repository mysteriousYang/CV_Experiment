# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ratio = 0.7

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)#这里本来是个True
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=False)#这里本来是个True
    
    def forward(self, x):
        identity = x.detach()
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity
        return self.relu(out)


class My_CNN(nn.Module):
    def __init__(self):
        super(My_CNN,self).__init__()
        self.conv1 = ConvBlock(in_channels = 1, out_channels = 64, kernel_size = 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.layer1 = self._make_layer(in_channels = 64, out_channels = 64, blocks = 2)

        self.conv2 = ConvBlock(64, 16, 3)
        self.pool2 = nn.MaxPool2d(kernel_size = 2)

        self.fc1 = nn.Linear(16*13*11, 96)
        self.fc2 = nn.Linear(96, 64)
        self.fc3 = nn.Linear(64, 40)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        # print("conv1:  ", x.shape)
        x = self.pool1(x)
        # print("pool1: ", x.shape)
        x = self.layer1(x)
        # print("layer1: ", x.shape)
        x = self.conv2(x)
        # print("conv2: ", x.shape)
        x = self.pool2(x)
        # print("pool2: ", x.shape)

        x = x.view(-1, 16*13*11)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def _load_face_dataset():
     # 获取代码文件的绝对路径
    file_dir = os.path.dirname(os.path.abspath(__file__))

    # 设置当前工作目录为代码文件所在的目录
    os.chdir(file_dir)

    # 检查文件是否存在
    if not os.path.isfile('olivettifaces.png'):
        print("File not found")
    
    image = cv.imread('olivettifaces.png')
    # 判断是否成功读取图像
    if image is None:
        print('Failed to read olivettifaces.png.')
        sys.exit()

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cells = [np.hsplit(row, 20) for row in np.vsplit(gray, 20)]
    x = np.array(cells)
    #由于47是质数，不适合后续池化，需要添加一列空白数据
    # edge = np.zeros(57)
    # x = np.insert(x,47,edge,axis=3)
    #print(x)

    data = x.reshape(-1, 1, 57, 47).astype(np.float32) 
    k = np.arange(40) 
    labels = np.repeat(k, 10)  
    
    # 将numpy的ndarray转换为Torch的Tensor结构
    data = torch.tensor(data).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)

    dataset = TensorDataset(data, labels)
    
    # 按照7：3的比例划分训练集和验证集
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # 用DataLoader对象将训练集和测试集组织成不同批次（batch）
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, test_loader

def run():

    train_loader, test_loader = _load_face_dataset()
    model = My_CNN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()  # 定义分类损失函数为交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，将学习率设定为0.001

    
    # 训练阶段
    model.train()  # 设定模型为训练模式
    num_epochs = 10  # 总共训练多少轮
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs:torch.Tensor = inputs.to(device)
            labels:torch.Tensor = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)             # [batch_size, in_channels, H, W]
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



if __name__ == "__main__":

    run()