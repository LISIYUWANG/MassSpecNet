# -*- coding: utf-8 -*-
# @Time    : 2022/1/29 0:01
# @Author  : naptmn
# @File    : compare_fig2.py
# @Software: PyCharm
import matplotlib.pyplot as plt

# 数据 如果有时间可以帮我核对一下。。我从excel里面粘出来的
# data begin
# MassSpecNet: acc:0.87272727 auc:0.9313
# Ghostnet: acc:0.82181818 auc:0.9096
# Mobilenetv2: acc:0.88 auc:0.9141
# data end

if __name__ =='__main__':
    # 因为数据比较少 所以这里就直接手敲进去数据了
    label = ["MsNet", "Ghostnet", "Mobilenet_v2","mobilenet_v3_small"]
    acc = [0.87264, 0.82181818, 0.88,0.836]
    auc = [0.9288, 0.9096, 0.9141,0.904]
    fig, axes = plt.subplots(1, 1, figsize=(8, 4))
    axes.plot(label, acc, linestyle='-', label="Accuracy", color="#845EC2", marker='x', linewidth=3)
    axes.plot(label, auc, linestyle='-', label="Area under curve", color="#2E839E", marker='o', linewidth=3)
    # y轴坐标
    axes.set_ylabel("Accuracy/Area under curve")
    # x轴坐标
    axes.set_xlabel("Net")
    axes.legend()
    plt.savefig(fname='./figures/fig2.svg', format='svg')
    plt.show()