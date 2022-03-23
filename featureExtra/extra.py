# -*- coding: utf-8 -*-
# @Time    : 2022/1/13 16:52
# @Author  : naptmn
# @File    : extra.py
# @Software: PyCharm
import pickle

import torch
from model.CEEnet.best import CEEnetNet2rui
from model.ghostnet.GhostNet import ghostnet
from torchvision import datasets, transforms ,models
import torchvision.models as models
from model.CEEnet.net import NetConcat
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
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
    NAME ='data7_3_orgin'  #'PXD007088'#'PXD008383'#'photoss'#

    BATCH_SIZE = 1
    RANDOMSEED = 620664
    seed_torch(RANDOMSEED)
    # 提取特征的网络
    # model = NetConcat()
    # model.load_state_dict(torch.load('../test/net/NetConcat_Simple_proce_fg_data7_3_orgin.pth'))

    # model = models.mobilenet_v3_small(num_classes = 2)
    # model.load_state_dict(torch.load('../test/net/data7_3_orgin_mobilenet_v3_small.pth'))
    # model = model.features
    #model = models.mobilenet_v2(num_classes = 2)
    #model.load_state_dict(torch.load('../test/net/data7_3_orgin_mobilenet_v2.pth'))
    #model = model.features
    model = ghostnet()
    model.load_state_dict(torch.load('../test/net/data7_3_orgin_ghostnet.pth'))
    print(model)
    # 设置参数不更新梯度
    for param in model.parameters():
        param.requires_grad = False
    #print(model)
    #------------------------------------------------------------------------------------------------------------------------
    train_set = datasets.ImageFolder('../data/' + NAME + '/trainset',
                                                 transform=transforms.Compose([
                                                     transforms.Resize(256),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                                 )
    val_set = datasets.ImageFolder('../data/' + NAME + '/testset',
                                               transform=transforms.Compose([
                                                   transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                               )
    len = len(train_set)
    torch.manual_seed(RANDOMSEED)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    torch.manual_seed(RANDOMSEED)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    #--------------------------------------------------------------------------------------------------------------------
    # data_loader = datasets.ImageFolder('../data/'+NAME,
    #                                             transform=transforms.Compose([
    #                                                 transforms.Resize(256),
    #                                                 transforms.CenterCrop(224),
    #                                                 transforms.ToTensor(),
    #                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #                                             )
    #
    # # 读取标签
    # label = []
    # for i , data in enumerate(data_loader, 0 ):
    #     _, label_ = data
    #     label.append(label_)
    # train_set , test_set = train_test_split(data_loader,test_size=0.3,random_state=RANDOMSEED,stratify=label)
    # len = len(train_set)
    # torch.manual_seed(RANDOMSEED)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    # torch.manual_seed(RANDOMSEED)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    #------------------------------------------------------------------------------------------------------------------------------


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    # feature不同 但是label是相同的
    # CEEnetNet2rui
    feature_train_fc1_before = []
    feature_train_fc2_before = []
    feature_train_fc3_before = []
    feature_train_fc3_after = []
    labels_train = []
    feature_test_fc1_before = []
    feature_test_fc2_before = []
    feature_test_fc3_before = []
    feature_test_fc3_after = []
    labels_test= []
    # ghostnet
    feature_test_dropoutBefore= []
    feature_test_dropoutAfter= []
    feature_train_dropoutBefore= []
    feature_train_dropoutAfter= []
    # mobilenet_v2
    feature_train_ConvBNActivationV2 = []
    feature_test_ConvBNActivationV2 = []
    # mobilenet_v3
    feature_train_ConvBNActivationV3 = []
    feature_test_ConvBNActivationV3 = []
    model.eval()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        labels_train.append(labels.cpu().numpy().squeeze())
        # CEEnetNet2rui
        # feature_train_fc1_before.append(model.fc1before.cpu().numpy().squeeze())
        # feature_train_fc2_before.append(model.fc2before.cpu().numpy().squeeze())
        # feature_train_fc3_before.append(model.fc3before.cpu().numpy().squeeze())
        # feature_train_fc3_after.append(model.fc3after.cpu().numpy().squeeze())
        # # ghostnet
        #feature_train_dropoutBefore.append(model.dropoutBefore.cpu().numpy().squeeze())
        #print('shape: ',model.dropoutBefore.cpu().numpy().squeeze().shape)
        #feature_train_dropoutAfter.append(model.dropoutBefore.cpu().numpy().squeeze())
        # mobilenetv2
        feature_train_ConvBNActivationV2.append(outputs.cpu().numpy().squeeze())
        print('shape: ',outputs.cpu().numpy().squeeze().shape)

        # print('dd ',type(model.features.cpu()))
        # print('dd ', model.features.cpu())
        # mobilenetv3
        #feature_train_ConvBNActivationV3.append(outputs.cpu().numpy().squeeze())
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        #outputs = outputs.view(outputs.size(0), -1)
        labels_test.append(labels.cpu().numpy().squeeze())
        # CEEnetNet2rui
        # feature_test_fc1_before.append(model.fc1before.cpu().numpy().squeeze())
        # feature_test_fc2_before.append(model.fc2before.cpu().numpy().squeeze())
        # feature_test_fc3_before.append(model.fc3before.cpu().numpy().squeeze())
        # feature_test_fc3_after.append(model.fc3after.cpu().numpy().squeeze())
        # # # ghostnet
        #feature_test_dropoutBefore.append(model.dropoutBefore.cpu().numpy().squeeze())
        #feature_test_dropoutAfter.append(model.dropoutBefore.cpu().numpy().squeeze())
        # # mobilenet_v2
        #feature_test_ConvBNActivationV2.append(outputs.cpu().numpy().squeeze())
        # # mobilenet_v3
        #feature_test_ConvBNActivationV3.append(outputs.cpu().numpy().squeeze())
    #print(np.array(feature_test_ConvBNActivationV3).shape)
    # print(np.array(feature_train_fc3_before).shape)
    # # print(np.array(labels_train).shape)
    # print(np.array(feature_test_fc2_before).shape)
    # print(np.array(feature_test_fc3_before).shape)
    # print(np.array(labels_test).shape)
    print('----Saving features----')
    pickle.dump(labels_train, open('./features/labels_train' + NAME + 'n.pkl', 'wb'))
    pickle.dump(labels_test, open('./features/labels_test' + NAME + 'n.pkl', 'wb'))
    # CEEnetNet2rui
    # pickle.dump(feature_train_fc1_before, open('./features/feature_train_fc1_before'+NAME+'_trans.pkl', 'wb'))
    # pickle.dump(feature_train_fc2_before, open('./features/feature_train_fc2_before'+NAME+'_trans.pkl', 'wb'))
    # pickle.dump(feature_train_fc3_before, open('./features/feature_train_fc3_before'+NAME+'_trans.pkl', 'wb'))
    # pickle.dump(feature_train_fc3_after, open('./features/feature_train_fc3_after'+NAME+'_trans.pkl', 'wb'))
    # pickle.dump(feature_test_fc1_before, open('./features/feature_test_fc1_before'+NAME+'_trans.pkl', 'wb'))
    # pickle.dump(feature_test_fc2_before, open('./features/feature_test_fc2_before'+NAME+'_trans.pkl', 'wb'))
    # pickle.dump(feature_test_fc3_before, open('./features/feature_test_fc3_before'+NAME+'_trans.pkl', 'wb'))
    # pickle.dump(feature_test_fc3_after, open('./features/feature_test_fc3_after'+NAME+'_trans.pkl', 'wb'))
    # # ghostnet
    #pickle.dump(feature_train_dropoutBefore, open('./features/feature_train_dropoutBefore' + NAME + '_trans.pkl', 'wb'))
    #pickle.dump(feature_train_dropoutAfter, open('./features/feature_train_dropoutAfter' + NAME + '_trans.pkl', 'wb'))
    #pickle.dump(feature_test_dropoutBefore, open('./features/feature_test_dropoutBefore' + NAME + '_trans.pkl', 'wb'))
    #pickle.dump(feature_test_dropoutAfter, open('./features/feature_test_dropoutAfter' + NAME + '_trans.pkl', 'wb'))
    # # mobilenet_v2
    #pickle.dump(feature_train_ConvBNActivationV2, open('./features/feature_train_ConvBNActivationV2' + NAME + '_trans.pkl', 'wb'))
    #pickle.dump(feature_test_ConvBNActivationV2, open('./features/feature_test_ConvBNActivationV2' + NAME + '_trans.pkl', 'wb'))
    # # mobilenet_v3Small
    #pickle.dump(feature_train_ConvBNActivationV3, open('./features/feature_train_ConvBNActivationV3' + NAME + '_trans.pkl', 'wb'))
    #pickle.dump(feature_test_ConvBNActivationV3, open('./features/feature_test_ConvBNActivationV3' + NAME + '_trans.pkl', 'wb'))
    print('----Finish saving features----')
