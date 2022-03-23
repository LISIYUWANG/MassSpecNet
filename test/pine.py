import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
from torchvision import datasets, models, transforms
from model.CEEnet.best import NetConcat as Cee
from model.CEEnet.best import CEEnetNorui,CEEnetNet1rui,CEEnetNet2rui
from model.CEEnet.best import ghostnet as ComHlf
from model.ghostnet.GhostNet import ghostnet as Ghost
from model.mobilenet.mobilenetv3 import MobileNetV3_Small as Mobile
import torchvision.models as models
from sklearn.metrics import roc_auc_score,f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
def draw(acc,auc,f1,loss,f_acc,f_auc,f_f1,name):
    x = np.arange(REALEPOCH+1)
    plt.figure(figsize=(10, 10))
    #plt.subtitle(NAME)
    plt.subplot(2,2,1)
    plt.plot(x,loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss on train'+'('+NAME+'_'+name+')')
    plt.subplot(2,2,2)
    plt.plot(x,auc)
    plt.xlabel('Epoch'+'   '+str(f_auc))
    plt.ylabel('Auc')
    plt.title('Auc on test'+'('+NAME+'_'+name+')')
    plt.subplot(2,2,3)
    plt.plot(x,f1)
    plt.xlabel('Epoch'+'   '+str(f_f1))
    plt.ylabel('F1 score')
    plt.title('F1 Score on test'+'('+NAME+'_'+name+')')
    plt.subplot(2,2,4)
    plt.plot(x,acc)
    plt.xlabel('Epoch'+'   '+str(f_acc))
    plt.ylabel('Acc')
    plt.title('Acc on test'+'('+NAME+'_'+name+')')
    plt.savefig('./result/'+NAME+'_'+name+'.svg', format='svg')
    plt.show()

def write(acc,auc,f1,loss):
    # x = np.arange(REALEPOCH + 1)
    output = open(NAME+'_'+name+'.csv', 'w')
    output.write('acc')
    output.write(',')
    output.write('auc')
    output.write(',')
    output.write('f1')
    output.write(',')
    output.write('loss')
    output.write('\n')
    for i in range(REALEPOCH + 1):
        output.write(str(acc[i]))
        output.write(',')
        output.write(str(auc[i]))
        output.write(',')
        output.write(str(f1[i]))
        output.write(',')
        output.write(str(loss[i]))
        output.write('\n')
if __name__ =="__main__":

    EPOCHMAX = 500  # 设置最大迭代次数
    BATCH_SIZE = 4
    LR = 0.01  # learning rate
    RANDOMSEED = 620664
    REALEPOCH=0
    #DATASET = os.listdir('./data')
    NAME ='data7_3_orgin'#'PXD008383' #'PXD007088'##
    GPU = torch.cuda.is_available()
    print(GPU)
    seed_torch(RANDOMSEED)
    model_ls = {}
    #model_name_ls = ['CEEnetDelChange','Moblienet','Ghostnet','ComplexHalf','focSeNet','CEEnet']
    model_name_ls = ['CEEnetNet2rui'] # ,'mobilenet_v3_small','CEEnetNet2rui','ghostnet'
    #model_name_ls = ['CEEnetDelChangeNorui', 'CEEnetDelChangeNet1rui', 'CEEnetDelChangeNet2rui']
    #model_ls['CEEnet'] = Cee() #网络1：完整网络，注意力类型CA
    # model_ls['focSeNet']= Cee(foc_type=2) #网络2：完整网络，注意力类型SE
    # model_ls['ComplexHalf'] = ComHlf(if_conc=False,foc_type=1)#网络3：完整网络的复杂部分
    # model_ls['Ghostnet'] = Ghost()#网络4：ghostnet
    # model_ls['Moblienet'] = Mobile()#网络5：mobilenet
    # model_ls['CEEnetDelChange'] = Cee(del_change_conv=True) #网络6：无可变形卷积
    # model_ls['CEEnetNorui'] = CEEnetNorui(foc_type=1)
    # model_ls['CEEnetNet1rui'] = CEEnetNet1rui(foc_type=1)
    model_ls['CEEnetNet2rui'] = CEEnetNet2rui(foc_type=1)
    # model_ls['CEEnetDelChangeNorui'] = CEEnetNorui(del_change_conv=True)
    # model_ls['CEEnetDelChangeNet1rui'] = CEEnetNet1rui(del_change_conv=True)
    # model_ls['CEEnetDelChangeNet2rui'] = CEEnetNet2rui(del_change_conv=True)
    #model_ls['mobilenet_v2'] = models.mobilenet_v2(num_classes=2)
    #model_ls['mobilenet_v3_small'] = models.mobilenet_v3_small(num_classes=2)
    #model_ls['ghostnet'] = Ghost()
    # model_ls['mobilenet_v3_large'] = models.mobilenet_v3_large(num_classes=2)
    # model_ls['resnet34'] = models.resnet34(num_classes=2)
    # model_ls['resnet18'] = models.resnet18(num_classes=2)
    # 第二类测试 CEEnet与CEEnetDelChangenet
    # 画图
    # model_name_ls = ['Ghostnet', 'CEEnetNet2rui']
    # model_name_ls = [ 'CEEnetNet2rui']
    #model_name_ls = ['Moblienetv2']
#------------------------------------------------------------------------------------------------------------------------
    train_set = torchvision.datasets.ImageFolder('../data/'+NAME+'/trainset',
                                                 transform=transforms.Compose([
                                                     transforms.Resize(256),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                                 )
    val_set = torchvision.datasets.ImageFolder('../data/'+NAME+'/testset',
                                                transform=transforms.Compose([
                                                   transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                               )
    len = len(train_set)
    torch.manual_seed(RANDOMSEED)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,num_workers = 2)
    torch.manual_seed(RANDOMSEED)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True,num_workers = 2)
    seed_torch(RANDOMSEED)

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
    for name in model_name_ls:
        REALEPOCH = 0
        net = model_ls[name]
        test_x = []
        test_y = []
        # 是否使用gpu运算
        if GPU:
            net = net.cuda()
        critterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD([{'params': net.parameters(), 'initial_lr': LR}], lr=LR, momentum=0.9)
        # 自动调整学习率
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20,gamma=0.8)
        auc = []
        acc = []
        f1 = []
        loss_all = []
        acc_avg=0
        auc_avg=0
        f1_avg=0
        loss_avg=0
        f_auc = 0
        f_acc = 0
        f_f1 = 0
        loss_num = 0
        for epoch in range(EPOCHMAX):
            if epoch > 300:
                break
            running_loss = 0.0
            net.train()
            for i , data in enumerate(train_loader, 0 ):
                inputs, labels = data
                if GPU:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                # inputs = inputs[:,1,:]
                # inputs = inputs.unsqueeze(1)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = critterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            #print('[%d] lr: %.4f'% (epoch+1,scheduler.get_last_lr()[0]))
            print('[%d] loss: %.3f'% (epoch+1,running_loss/len))
            if(running_loss == 0):
                loss_num += 1
            else:
                loss_num = 0
            if loss_num == 10:
                break
            loss_all.append(running_loss/len)
            loss_avg +=running_loss/len
            correct = 0
            total = 0
            lable = []
            prob_f1 = []
            prob_auc = []
            net.eval()
            for _,data in enumerate(test_loader, 0 ):
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
                        #print(predicted,labels)
                        correct += (predicted == labels).sum().item()


            lable = [i.cpu() for i in lable]
            lable = np.array(lable).astype(int)

            auc_avg+=roc_auc_score(lable, np.array(prob_auc))
            acc_avg+=100 * correct / total
            f1_avg+=f1_score(lable, np.array(prob_f1))
            print("AUC:{:.4f}".format(roc_auc_score(lable, np.array(prob_auc))))
            print("F1-Score:{:.4f}".format(f1_score(lable, np.array(prob_f1))))
            print('Accuracy of the network on the 7:3 test images: %f %%' % (100 * correct / total))
            auc.append(roc_auc_score(lable, np.array(prob_auc)))
            f1.append(f1_score(lable, np.array(prob_f1)))
            acc.append(correct / total)
            if((epoch+1)%10==0):#每10论计算一下 是否相较上次个相差在0.1  是的话停止
                auc_avg = auc_avg/10
                acc_avg = acc_avg/10
                f1_avg = f1_avg/10
                loss_avg = loss_avg/10
                if(abs(roc_auc_score(lable, np.array(prob_auc))-auc_avg)<0.001 and abs(f1_score(lable, np.array(prob_f1))-f1_avg)<0.001 and abs(running_loss/len-loss_avg)<0.001 ):
                    f_acc = correct / total
                    f_auc = roc_auc_score(lable, np.array(prob_auc))
                    f_f1 = f1_score(lable, np.array(prob_f1))
                    break
                if ((epoch + 1) == EPOCHMAX):
                    print("未达收敛条件，结束训练")
                    break  # 强行结束
                else:
                    acc_avg=0
                    auc_avg=0
                    f1_avg=0
                    loss_avg=0
            REALEPOCH+=1
            running_loss = 0.0
        #plt.plot(np.arange(epoch),auc,)
        torch.save(net.state_dict(), 'net/' + NAME + '_' + name + '.pth')
        write(acc, auc, f1, loss_all)
        draw(acc,auc,f1,loss_all,f_acc,f_auc,f_f1,name)


        print('Finished Training')
