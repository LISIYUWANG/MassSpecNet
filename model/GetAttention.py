# 首先需要安装Evison
#!pip install Evison

from Evison import Display, show_network
from torchvision import models
import torch
from model.CEEnet.net2 import NetConcat
from torchvision import datasets, transforms ,models
import torchvision.models as models
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from model.ghostnet.GhostNet import ghostnet
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
    NAME = 'data7_3_orgin'  #'photoss' #

    BATCH_SIZE = 1
    RANDOMSEED = 620664
    seed_torch(RANDOMSEED)
    # 生成我们需要可视化的网络(可以使用自己设计的网络)
    #network = models.efficientnet_b0(pretrained=True)
    # network = CEEnetNet2rui(foc_type=1)
    # network.load_state_dict(torch.load('../test/net/data7_3_orgin_CEEnetNet2rui.pth'))
    network = NetConcat()
    network.load_state_dict(torch.load('../test/net/NetConcat_Simple_proce_fg_data7_3_orgin.pth'))
    # network = models.mobilenet_v3_small(num_classes = 2)
    # network.load_state_dict(torch.load('../test/net/data7_3_orgin_mobilenet_v3_small.pth'))
    # network = models.mobilenet_v2(num_classes = 2)
    # network.load_state_dict(torch.load('../test/net/data7_3_orgin_mobilenetv2.pth'))
    # network = ghostnet()
    # network.load_state_dict(torch.load('../test/net/data7_3_orgin_Ghostnet.pth'))
    # 使用show_network这个辅助函数来看看有什么网络层(layers)
    show_network(network)


    '''
    net1.blocks.4.0.ca
    net1.blocks.7.0.ca
    net2.ca
    
    'blocks.8.1.se'
    'blocks.7.0.se'
    'blocks.6.4.se'
    'blocks.6.3.se'
    'blocks.4.0.se'
    'blocks.3.0.se'
    '''
    # 构建visualization的对象 以及 制定可视化的网络层
    visualized_layer = 'net1.blocks.4.0.ca'
    display = Display(network, visualized_layer,norm=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), img_size=(224, 224))  # img_size的参数指的是输入图片的大小


    # # 加载我们想要可视化的图片
    # from PIL import Image
    # image = Image.open('Dog_and_cat.jpeg').resize((224, 224))
    #
    # # 将想要可视化的图片送入display中，然后进行保存
    # display.save(image)
    #
    #
    #
    # data_loader = datasets.ImageFolder('../data/'+NAME,
    #                                             transform=transforms.Compose([
    #                                                 transforms.Resize(256),
    #                                                 transforms.CenterCrop(224),
    #                                                 transforms.ToTensor(),
    #                                                 #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                                             ])
    #                                             )
    #
    # test_loader = torch.utils.data.DataLoader(data_loader, batch_size=BATCH_SIZE, shuffle=True)
    # for i,(imges,labels) in enumerate(test_loader):
    #     print(imges.shape)
    #     print(type(imges))
    #     # #labels = imges.to(device)
    #     # outputs = model(imges)
    #     # print(outputs.shape)
    #     # print(type(outputs))
    #     # break
    #     imges=np.array(imges)
    #     imges=torch.from_numpy(imges)
    #     for j in range(len(imges)):
    #         imge = imges[j]
    #         imge = (imge).numpy().transpose(1,2,0)
    #         print('imge',imge.shape)
    #         print(type(imge))
    #         # from PIL import Image
    #         # imge = Image.fromarray(np.uint8(imge))
    #         # imge.show()
    #         # display.save(imge)
    #
    #         import matplotlib.pyplot as plt
    #         plt.imshow(imge)
    #         plt.axis('off')
    #         plt.title(labels[j])
    #         plt.show()
    #         break
    from PIL import Image
    # image2 = Image.open('..\\data\data7_3_orgin\\testset\\N\\guot_PC1_170127_CPP114_sw.mzXML.gz.image.0.itms.png')
    # Resize = transforms.Resize(256)
    # CenterCrop = transforms.CenterCrop(224)
    # image2 = Resize(image2)
    # image2 = CenterCrop(image2)
    # #image2.show()
    # display.save(image2,file=visualized_layer)

    from os.path import join as pjoin
    path = '..\\data\\data7_3_orgin\\testset'
    path = '../data/PXD007088'
    k = 0
    kk = 0
    for i in os.listdir(path):  # os.listdir(path_old) 获取该文件夹下的文件名
        person_dir = pjoin(path, i)
        print(i)
        # kk += 1
        # if kk == 1:
        #     continue
        k = 0
        for j in os.listdir(person_dir):
            k += 1
            if k > 10:
                break
            path2 = pjoin(person_dir, j)
            name = j.split('.')[0]
            name = i+'_'+name
            #print(name)
            image2 = Image.open(path2)
            Resize = transforms.Resize(256)
            CenterCrop = transforms.CenterCrop(224)
            image2 = Resize(image2)
            image2 = CenterCrop(image2)
            display.save(image2,file=name)