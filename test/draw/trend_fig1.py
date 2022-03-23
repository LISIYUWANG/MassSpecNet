# -*- coding: utf-8 -*-
# @Time    : 2022/1/28 21:11
# @Author  : naptmn
# @File    : trend_fig1.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def drawacc():
    # 用户画出第一列1x3的三个图 为acc与loss之间的关系
    path1 = '../data7_3_orgin_4_109.csv'
    path2 = '../Ghostnet.csv'
    path3 = '../Moblienetv2.csv' # 这里路径名称没错 是我代码里面的名称拼错了
    path4 = '../mobilenet_v3_small.csv'
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)
    df4 = pd.read_csv(path4)
    acc1 = df1['acc'].tolist()
    loss1 = df1['loss'].tolist()
    acc2 = df2['acc'].tolist()
    loss2 = df2['loss'].tolist()
    acc3 = df3['acc'].tolist()[:201]
    loss3 = df3['loss'].tolist()[:201]
    acc4 = df4['acc'].tolist()
    loss4 = df4['loss'].tolist()
    # 这里是网络名称 可以在这里改
    name1 = 'MsNet'
    name2 = 'Ghostnet'
    name3 = 'Mobilenet_v2'
    name4 = 'mobilenet_v3_small'
    # 画布大小 根据实际情况调整
    plt.figure(figsize=(15, 5)) # 第一个参数为宽 第二个为高
    plt.subplot(1,4,1) # 第一个图
    # 这里是标题
    plt.title(name1)
    l1, = plt.plot(range(len(acc1)),acc1 )
    l2, = plt.plot(range(len(loss1)),loss1 )
    # 图例
    plt.legend(handles=[l1, l2], labels=['Accuracy', 'Loss'], loc='center right')
    plt.xlabel('iterations')
    plt.ylabel('Accuracy/Loss')

    plt.subplot(1, 4, 2)  # 第一个图
    plt.title(name2)
    l1, = plt.plot(range(len(acc2)),acc2)
    l2, = plt.plot(range(len(loss2)),loss2 )
    # 图例
    plt.legend(handles=[l1, l2], labels=['Accuracy', 'Loss'], loc='center right')
    plt.xlabel('iterations')
    plt.ylabel('Accuracy/Loss')

    plt.subplot(1, 4, 3)  # 第一个图
    plt.title(name3)
    l1, = plt.plot(range(len(acc3)),acc3)
    l2, = plt.plot(range(len(loss3)),loss3 )
    # 图例
    plt.legend(handles=[l1, l2], labels=['Accuracy', 'Loss'], loc='center right')
    plt.xlabel('iterations')
    plt.ylabel('Accuracy/Loss')

    plt.subplot(1, 4, 4)  # 第一个图
    plt.title(name4)
    l1, = plt.plot(range(len(acc4)),acc4)
    l2, = plt.plot(range(len(loss4)),loss4 )
    # 图例
    plt.legend(handles=[l1, l2], labels=['Accuracy', 'Loss'], loc='center right')
    plt.xlabel('iterations')
    plt.ylabel('Accuracy/Loss')

    plt.savefig('./figures/fig1_1.svg', format='svg')
    plt.show()

def drawauc():
    # 用户画出第二列1x3的三个图 为auc与loss之间的关系
    path1 = '../data7_3_orgin_4_109.csv'
    path2 = '../Ghostnet.csv'
    path3 = '../Moblienetv2.csv'  # 这里路径名称没错 是我代码里面的名称拼错了
    path4 = '../mobilenet_v3_small.csv'
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)
    df4 = pd.read_csv(path4)
    auc1 = df1['auc'].tolist()
    loss1 = df1['loss'].tolist()
    auc2 = df2['auc'].tolist()
    loss2 = df2['loss'].tolist()
    auc3 = df3['auc'].tolist()[:201]
    loss3 = df3['loss'].tolist()[:201]
    auc4 = df4['auc'].tolist()
    loss4 = df4['loss'].tolist()
    # 这里是网络名称 可以在这里改
    name1 = 'MsNet'
    name2 = 'Ghostnet'
    name3 = 'Mobilenet_v2'
    name4 = 'mobilenet_v3_small'
    # 画布大小 根据实际情况调整
    plt.figure(figsize=(15, 5))  # 第一个参数为宽 第二个为高
    plt.subplot(1, 4, 1)  # 第一个图
    # 这里是标题
    plt.title(name1)
    l1, = plt.plot(range(len(auc1)), auc1)
    l2, = plt.plot(range(len(loss1)), loss1)
    # 图例
    plt.legend(handles=[l1, l2], labels=['Area under curve', 'Loss'], loc='center right')
    plt.xlabel('iterations')
    plt.ylabel('Area under curve/Loss')

    plt.subplot(1, 4, 2)  # 第一个图
    plt.title(name2)
    l1, = plt.plot(range(len(auc2)), auc2)
    l2, = plt.plot(range(len(loss2)), loss2)
    # 图例
    plt.legend(handles=[l1, l2], labels=['Area under curve', 'Loss'], loc='center right')
    plt.xlabel('iterations')
    plt.ylabel('Area under curve/Loss')

    plt.subplot(1, 4, 3)  # 第一个图
    plt.title(name3)
    l1, = plt.plot(range(len(auc3)), auc3)
    l2, = plt.plot(range(len(loss3)), loss3)
    # 图例
    plt.legend(handles=[l1, l2], labels=['Area under curve', 'Loss'], loc='center right')
    plt.xlabel('iterations')
    plt.ylabel('Area under curve/Loss')

    plt.subplot(1, 4, 4)  # 第一个图
    plt.title(name4)
    l1, = plt.plot(range(len(auc4)), auc4)
    l2, = plt.plot(range(len(loss4)), loss4)
    # 图例
    plt.legend(handles=[l1, l2], labels=['Area under curve', 'Loss'], loc='center right')
    plt.xlabel('iterations')
    plt.ylabel('Area under curve/Loss')

    plt.savefig('./figures/fig1_2.svg', format='svg')
    plt.show()
if __name__ =='__main__':
    drawacc()
    drawauc()
