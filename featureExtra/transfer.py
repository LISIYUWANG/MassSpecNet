# -*- coding: utf-8 -*-
# @Time    : 2022/1/13 16:52
# @Author  : naptmn
# @File    : extra.py
# @Software: PyCharm
import pickle

import torch
from model.CEEnet.best import CEEnetNet2rui
from torchvision import datasets, transforms ,models
import torchvision.models as models
import os
import random
from sklearn.metrics import roc_auc_score,f1_score
import numpy as np
def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    NAME = 'data7_3_orgin'
    BATCH_SIZE = 4
    RANDOMSEED = 620664
    seed_torch(RANDOMSEED)
    NAME = 'data7_3_orgin_rui7'  # 84
    GPU = torch.cuda.is_available()
    print(GPU)

    # 提取特征的网络
    net = CEEnetNet2rui(foc_type=1)
    net.load_state_dict(torch.load('../test/net/data7_3_orgin_CEEnetNet2rui.pth'))
    # net = models.mobilenet_v2()
    # net.load_state_dict(torch.load('../test/net/data7_3_orgin_Moblienetv2.pth'))

    val_set = datasets.ImageFolder('../data/' + NAME ,
                                               transform=transforms.Compose([
                                                   transforms.Resize(256),

                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                               )
    len = len(val_set)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_x = []
    test_y = []
    # 是否使用gpu运算
    if GPU:
        net = net.cuda()
    auc = []
    acc = []
    f1 = []
    loss_all = []
    acc_avg = 0
    auc_avg = 0
    f1_avg = 0
    loss_avg = 0
    f_auc = 0
    f_acc = 0
    f_f1 = 0
    correct = 0
    total = 0
    lable = []
    prob_f1 = []
    prob_auc = []
    for _, data in enumerate(test_loader, 0):
        with torch.no_grad():
            images, labels = data
            # images = images[:, 1, :]
            # images = images.unsqueeze(1)
            if GPU:
                images = images.cuda()
                labels = labels.cuda()
                outputs = net(images)
                prob = outputs
                prob = prob.cpu().numpy()  # 先把prob转到CPU上，然后再转成numpy，如果本身在CPU上训练的话就不用先转成CPU了
                prob_auc.extend(prob[:, 1])
                prob_f1.extend(np.argmax(prob, axis=1))  # 求每一行的最大值索引
                lable.extend(labels)
            if GPU:
                outputs = outputs.cuda()
                _, predicted = torch.max(outputs.data, 1)
            if GPU:
                predicted = predicted.cuda()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    lable = [i.cpu() for i in lable]
    lable = np.array(lable).astype(int)

    auc_avg += roc_auc_score(lable, np.array(prob_auc))
    acc_avg += 100 * correct / total
    f1_avg += f1_score(lable, np.array(prob_f1))
    print("AUC:{:.4f}".format(roc_auc_score(lable, np.array(prob_auc))))
    print("F1-Score:{:.4f}".format(f1_score(lable, np.array(prob_f1))))
    print('Accuracy of the network on the 7:3 test images: %f %%' % (100 * correct / total))
    auc.append(roc_auc_score(lable, np.array(prob_auc)))
    f1.append(f1_score(lable, np.array(prob_f1)))
    acc.append(correct / total)

    print('Finished')
