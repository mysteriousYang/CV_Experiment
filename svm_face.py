# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from build_dataset import olivetti_face

if __name__ == '__main__':
    X, Y = olivetti_face()

    # 划分数据为训练集和测试集（7:3比例）
    train_data, test_data, train_labels, test_labels = train_test_split(X, Y, test_size=0.3, random_state=42)

    model = SVC(kernel="sigmoid")
    model.fit(train_data,train_labels)
    labels_pred = model.predict(test_data)

    print(classification_report(labels_pred, test_labels))
    