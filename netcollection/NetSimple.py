# -*- coding: utf-8 -*-
# @Time    : 2022/1/10 0:20
# @Author  : naptmn
# @File    : NetSimple.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
from torchvision import transforms
from sklearn.metrics import roc_auc_score,f1_score
import numpy as np
import os
import random


#NetSimple的测试函数
# 固定部分（用于确认结果）
def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

if __name__ =='__main__':
    EPOCHMAX = 200  # 设置最大迭代次数
    BATCH_SIZE = 4
    LR = 0.01  # learning rate
    RANDOMSEED = 620664
    REALEPOCH = 0
    NAME = 'data7_3_orgin'
    GPU = torch.cuda.is_available()
    print(GPU)
    seed_torch(RANDOMSEED)

    train_set = torchvision.datasets.ImageFolder('./data/' + NAME + '/trainset',
                                                 transform=transforms.Compose([
                                                     transforms.Resize(256),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                                 )
    val_set = torchvision.datasets.ImageFolder('./data/' + NAME + '/testset',
                                               transform=transforms.Compose([
                                                   transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                               )
    len = len(train_set)
    torch.manual_seed(RANDOMSEED)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    torch.manual_seed(RANDOMSEED)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    seed_torch(RANDOMSEED)

    test_x = []
    test_y = []
    # 是否使用gpu运算
    net = Net()

    if GPU:
        net = net.cuda()
    critterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params': net.parameters(), 'initial_lr': LR}], lr=LR, momentum=0.9)
    # 自动调整学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.8)
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
    loss_num = 0
    for epoch in range(EPOCHMAX):
        running_loss = 0.0
        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            if GPU:
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = critterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len))
        loss_all.append(running_loss / len)
        loss_avg += running_loss / len
        correct = 0
        total = 0
        lable = []
        prob_f1 = []
        prob_auc = []
        net.eval()
        for _, data in enumerate(test_loader, 0):
            with torch.no_grad():
                images, labels = data
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
                    # print(predicted,labels)
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
        if ((epoch + 1) % 10 == 0):  # 每10论计算一下 是否相较上次个相差在0.1  是的话停止
            auc_avg = auc_avg / 10
            acc_avg = acc_avg / 10
            f1_avg = f1_avg / 10
            loss_avg = loss_avg / 10
            if (abs(roc_auc_score(lable, np.array(prob_auc)) - auc_avg) < 0.001 and abs(
                    f1_score(lable, np.array(prob_f1)) - f1_avg) < 0.001 and abs(
                    running_loss / len - loss_avg) < 0.001):
                f_acc = correct / total
                f_auc = roc_auc_score(lable, np.array(prob_auc))
                f_f1 = f1_score(lable, np.array(prob_f1))
                break
            else:
                acc_avg = 0
                auc_avg = 0
                f1_avg = 0
                loss_avg = 0
        REALEPOCH += 1
        running_loss = 0.0
    draw(acc, auc, f1, loss_all, f_acc, f_auc, f_f1)
    torch.save(net.state_dict(), 'net/NetConcat_Simple_proce_fg_' + NAME + '.pth')
    print('Finished Training')