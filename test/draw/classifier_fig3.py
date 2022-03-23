# -*- coding: utf-8 -*-
# @Time    : 2022/1/29 1:11
# @Author  : naptmn
# @File    : classifier_fig3.py
# @Software: PyCharm
# 这里我先按照我的想法画了 也即对我们网络的不同层输出（我记得之前跑实验的时候也是这个啊，而且我觉得目的是说明我们这个网络的端到端很方便，为什么需要别的网络）
# 因为和resnet,inception来不及跑了
import numpy as np
import matplotlib.pyplot as plt
def drawacc():
    # 4种类别
    label = ["Before fc1", "Before fc2", "Before fc3", "After fc3"]
    # 每个分类器对应四种类别的数据
    svc = [0.876363636, 0.865454545454545, 0.865454545454545, 0.865454545454545]
    rf = [0.869090909090909, 0.872727272727272, 0.872727272727272, 0.872727272727272]
    lr = [0.876363636363636, 0.872727272727272, 0.872727272727272, 0.872727272727272]
    xgboost = [0.861818181818181, 0.865454545454545, 0.872727272727272, 0.872727272727272]
    gaussiannb = [0.872727272727272, 0.847272727272727, 0.847272727272727, 0.847272727272727]

    # 开始画图
    x = np.arange(len(label))  # 标签位置
    width = 0.1  # 柱状图的宽度，可以根据自己的需求和审美来改
    fig, ax = plt.subplots()
    ax.bar(x - width * 2, svc, width=width, label="SVC", )
    ax.bar(x - width + 0.01, rf, width=width, label="RF", )
    ax.bar(x + 0.02, lr, width=width, label="LR", )
    ax.bar(x + width + 0.03, xgboost, width=width, label="XGBoost", )
    ax.bar(x + width * 2 + 0.04, gaussiannb, width=width, label="GaussianNB")
    ax.set_xticks(x)
    ax.set_xticklabels(label)
    ax.set_ylabel('Accuracy')
    # ax.set_xlabel('Features Layer')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    plt.savefig('./figures/fig3_1.svg', format='svg')
    plt.show()

def drawauc():
    label = ["Before fc1", "Before fc2", "Before fc3", "After fc3"]
    svc = [0.927589125145456, 0.92711308579287, 0.927060192531471, 0.92711308579287]
    rf = [0.923119644557283, 0.91513276208611, 0.898973870728869, 0.877578546493176]
    lr = [0.931291653443351, 0.931291653443351, 0.931291653443351, 0.931291653443351]
    xgboost = [0.861869247857823, 0.865439543002221, 0.872659473183116, 0.872659473183116]
    gaussiannb = [0.872633026552417, 0.84750872738813, 0.84750872738813, 0.930392467999576]
    x = np.arange(len(label))  # 标签位置
    width = 0.1  # 柱状图的宽度，可以根据自己的需求和审美来改
    fig, ax = plt.subplots()
    ax.bar(x - width * 2, svc, width=width, label="SVC", )
    ax.bar(x - width + 0.01, rf, width=width, label="RF", )
    ax.bar(x + 0.02, lr, width=width, label="LR", )
    ax.bar(x + width + 0.03, xgboost, width=width, label="XGBoost", )
    ax.bar(x + width * 2 + 0.04, gaussiannb, width=width, label="GaussianNB")
    ax.set_xticks(x)
    ax.set_xticklabels(label)
    ax.set_ylabel('Area under curve')
    # ax.set_xlabel('Features Layer')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    plt.savefig('./figures/fig3_2.svg', format='svg')
    plt.show()

def drawacc2():
    # 4种类别
    label = ['gaussiannb','lr','rf','svc','xgboost']
    # 每个分类器对应四种类别的数据
    #ms1
    # Mobilenet_v2=[0.862 ,0.858 ,0.865 ,0.865 ,0.815 ]
    # mobilenet_v3_small=[0.851 ,0.825 ,0.862 ,0.840 ,0.869 ]
    # Ghostnet_dropoutAfter=[0.818 ,0.815 ,0.836 ,0.818 ,0.829 ]
    # Ghostnet_dropoutBefore=[0.818 ,0.815 ,0.840 ,0.818 ,0.829 ]
    # MassSpecNet_fc1_before=[0.873,0.876,0.869,0.876,0.862]
    # MassSpecNet_fc2_before = [0.847,0.873,0.873,0.865,0.865]
    # MassSpecNet_fc3_after = [0.847,0.873,0.873,0.865,0.873]
    # MassSpecNet_fc3_before = [0.847 ,0.873 ,0.873 ,0.865 ,0.873 ]
    #PXD007088
    Mobilenet_v2=[0.705882353,0.529411765,0.647058824,0.588235294,0.529411765]
    mobilenet_v3_small=[0.647058824,0.411764706,0.647058824,0.588235294,0.529411765]
    Ghostnet_dropoutBefore=[0.647058824,0.647058824,0.588235294,0.705882353,0.470588235]
    MassSpecNet_fc1_before=[0.529411765,0.588235294,0.705882353,0.588235294,0.588235294]
    MassSpecNet_fc2_before = [0.705882353,0.588235294,0.411764706,0.588235294,0.411764706]
    MassSpecNet_fc3_before = [0.705882353,0.588235294,0.294117647,0.588235294,0.352941176]
    #PXD008383
    Mobilenet_v2=[0.888888889,0.888888889,0.888888889,0.888888889,0.944444444]
    mobilenet_v3_small=[0.777777778,0.833333333,0.833333333,0.833333333,0.777777778]
    Ghostnet_dropoutBefore=[0.277777778,0.444444444,0.277777778,0.277777778,0.444444444]
    MassSpecNet_fc1_before=[0.944444444,0.777777778,0.944444444,0.777777778,0.888888889]
    MassSpecNet_fc2_before = [0.611111111,0.555555556,0.333333333,0.444444444,0.722222222]
    MassSpecNet_fc3_before = [0.611111111,0.555555556,0.388888889,0.444444444,0.5]

    # 开始画图
    x = np.arange(len(label))  # 标签位置
    width = 0.1  # 柱状图的宽度，可以根据自己的需求和审美来改
    fig, ax = plt.subplots()
    ax.bar(x - width * 2, Mobilenet_v2, width=width, label="Mobilenet_v2", )
    ax.bar(x - width + 0.01, mobilenet_v3_small, width=width, label="mobilenet_v3_small", )
    ax.bar(x + 0.02, Ghostnet_dropoutBefore, width=width, label="Ghostnet", )
    #ax.bar(x + width + 0.03, Ghostnet_dropoutBefore, width=width, label="Ghostnet_dropoutBefore", )
    ax.bar(x + width * 1 + 0.03, MassSpecNet_fc1_before, width=width, label="MsNet_fc1_before")
    ax.bar(x + width * 2 + 0.04, MassSpecNet_fc2_before, width=width, label="MsNet_fc2_before", )
    #ax.bar(x + width * 3 + 0.05, MassSpecNet_fc3_after, width=width, label="MsNet_fc3_after", )
    ax.bar(x + width * 3 + 0.05, MassSpecNet_fc3_before, width=width, label="MsNet_fc3_before")

    ax.set_xticks(x)
    ax.set_xticklabels(label)
    ax.set_ylabel('Accuracy')
    # ax.set_xlabel('Features Layer')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=2)

    plt.ylim(ymax=1, ymin=0.2)
    plt.savefig('./figures/fig4_PXD008383_acc.svg', format='svg')
    plt.show()

def drawauc2():
    # 4种类别
    label = ['gaussiannb','lr','rf','svc','xgboost']
    # 每个分类器对应四种类别的数据
    #PXD007088
    Mobilenet_v2=[0.642857143,0.742857143,0.65,0.785714286,0.671428571]
    mobilenet_v3_small=[0.635714286,0.428571429,0.785714286,0.285714286,0.642857143]
    Ghostnet_dropoutBefore=[0.635714286,0.442857143,0.678571429,0.485714286,0.457142857]
    MassSpecNet_fc1_before=[0.514285714,0.642857143,0.664285714,0.471428571,0.657142857]
    MassSpecNet_fc2_before = [0.707142857,0.685714286,0.342857143,0.528571429,0.328571429]
    MassSpecNet_fc3_before = [0.685714286,0.685714286,0.242857143,0.5,0.4]
    #PXD008383
    Mobilenet_v2=[0.888888889,0.987654321,0.969135802,0.975308642,0.925925926]
    mobilenet_v3_small=[0.777777778,0.864197531,0.938271605,0.901234568,0.851851852]
    Ghostnet_dropoutBefore=[0.277777778,0.395061728,0.358024691,0.666666667,0.395061728]
    MassSpecNet_fc1_before=[0.944444444,0.802469136,0.975308642,0.790123457,0.919753086]
    MassSpecNet_fc2_before = [0.574074074,0.716049383,0.432098765,0.296296296,0.740740741]
    MassSpecNet_fc3_before = [0.574074074,0.716049383,0.49382716,0.703703704,0.530864198]
    # 开始画图
    x = np.arange(len(label))  # 标签位置
    width = 0.1  # 柱状图的宽度，可以根据自己的需求和审美来改
    fig, ax = plt.subplots()
    ax.bar(x - width * 2, Mobilenet_v2, width=width, label="Mobilenet_v2", )
    ax.bar(x - width + 0.01, mobilenet_v3_small, width=width, label="mobilenet_v3_small", )
    ax.bar(x + 0.02, Ghostnet_dropoutBefore, width=width, label="Ghostnet", )
    #ax.bar(x + width + 0.03, Ghostnet_dropoutBefore, width=width, label="Ghostnet_dropoutBefore", )
    ax.bar(x + width * 1 + 0.03, MassSpecNet_fc1_before, width=width, label="MsNet_fc1_before")
    ax.bar(x + width * 2 + 0.04, MassSpecNet_fc2_before, width=width, label="MsNet_fc2_before", )
    ax.bar(x + width * 3 + 0.05, MassSpecNet_fc3_before, width=width, label="MsNet_fc3_before", )
    #ax.bar(x + width * 4 + 0.06, MassSpecNet_fc3_before, width=width, label="MsNet_fc3_before")

    ax.set_xticks(x)
    ax.set_xticklabels(label)
    ax.set_ylabel('Area under curve')
    # ax.set_xlabel('Features Layer')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=2)
    plt.ylim(ymax=1, ymin=0.2)
    plt.savefig('./figures/fig4_PXD008383_auc.svg', format='svg')
    plt.show()
if __name__ =='__main__':
    drawacc2()
    drawauc2()
