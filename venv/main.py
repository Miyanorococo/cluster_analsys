#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np


def main():
    # クラスタ数
    N_CLUSTERS = 3
    s = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2])


    # base b
    # Blob データを生成する
    # dataset = datasets.make_blobs(centers=N_CLUSTERS)

    x5 = np.array([5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5])

    i = np.array([188.5,
                  239.2,
                  137.9,
                  207.3,
                  174.3,
                  224.3,
                  160.2,
                  340,
                  314.4,
                  349.2,
                  156.1,
                  84.6,
                  233.1,
                  81.5,
                  121.7,
                  2.4,
                  98.7,
                  98.4,
                  108,
                  98.8,
                  20.3,
                  163.7,
                  187,
                  41,
                  66,
                  103,
                  77,
                  137,
                  167,
                  125])

    dataset = np.dstack([x5, i, s])
    dataset = dataset.reshape([30, 3])
    print(dataset)


    # print(dataset[::-1, 0:3])

    # 特徴データ
    features = dataset[::-1, 0:2]
    # 正解ラベルは使わない
    # targets = dataset[1]

    print(features)

    # クラスタリングする
    cls = KMeans(n_clusters=N_CLUSTERS)
    pred = cls.fit_predict(features)

    # 各要素をラベルごとに色付けして表示する
    for i in range(N_CLUSTERS):
        if i == 0:
            col = 'Yellow'
            label0 = features[pred == i]
            plt.scatter(label0[:, 0], label0[:, 1], c=col)
        elif i == 1:
            col = 'Magenta'
            label1 = features[pred == i]
            plt.scatter(label1[:, 0], label1[:, 1], c=col)
        elif i == 2:
            col = 'Cyan'
            label2 = features[pred == i]
            plt.scatter(label2[:, 0], label2[:, 1], c=col)
        labels = features[pred]
        # plt.scatter(labels[:, 0], labels[:, 1], c=col)


    # クラスタのセントロイド (重心) を描く
    centers = cls.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], s=100,
                facecolors='none', edgecolors='black')

    #結果の表示
    labell = cls.labels_
    count = 0
    for i in range(dataset.shape[0]):
        if pred[i] == dataset[i, 2]:
            count += 1
        print(pred[i], dataset[i])
    print(label0)
    print(label1)
    print(label2)

    plt.show()


if __name__ == '__main__':
    main()