
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
   feature_name =[ 'ConvBNActivationV2','ConvBNActivationV3','dropoutAfter',
                   'dropoutBefore','fc1_before','fc2_before','fc3_after','fc3_before']
   classifier_name = ['gaussiannb','lr','rf','svc','xgboost']
   acc1 = [0.862 ,0.851 ,0.818 ,0.818 ,0.873 ,0.847 ,0.847 ,0.847 ,0.858
,0.825 ,0.815 ,0.815 ,0.876 ,0.873 ,0.873 ,0.873 ,0.865 ,0.862 ,0.836 ,0.840
,0.869 ,0.873 ,0.873 ,0.873 ,0.865 ,0.840 ,0.818 ,0.818 ,0.876 ,0.865 ,0.865
,0.865 ,0.815 ,0.869 ,0.829 ,0.829 ,0.862 ,0.865 ,0.873 ,0.873 ]
   auc1 =[0.862 ,0.851 ,0.818 ,0.818 ,0.873 ,0.848 ,0.930 ,0.848 ,0.941 ,0.907
,0.910 ,0.910 ,0.931 ,0.931 ,0.931 ,0.931 ,0.938 ,0.907 ,0.894 ,0.900 ,0.923 ,0.915
,0.878 ,0.899 ,0.937 ,0.905 ,0.907 ,0.907 ,0.928 ,0.927 ,0.927 ,0.927 ,0.814 ,0.869
,0.829 ,0.829 ,0.862 ,0.865 ,0.873 ,0.873 ]
   acc2=[0.870,0.870,0.826,0.826,0.739,0.913,0.391,0.913,0.522,0.522,0.522
,0.522,0.913,0.522,0.522,0.522,0.522,0.522,0.826,0.826,0.913,0.870,0.652,0.913,0.522,
         0.522,0.522,0.522,0.957,0.957,0.522,0.522,0.696,0.826,0.565,0.565,0.783
,0.870,0.565,0.826 ]
   auc2=[0.864 ,0.867 ,0.864 ,0.864 ,0.742 ,0.951 ,0.409 ,0.939 ,0.811 ,0.909
,0.833 ,0.833 ,0.977 ,0.962 ,0.311 ,0.962 ,0.500 ,0.500 ,0.886 ,0.917 ,0.977 ,0.985
,0.655 ,0.985 ,0.932 ,0.098 ,0.167 ,0.167 ,0.985 ,0.985, 0.659 ,0.500 ,0.686
,0.886 ,0.780 ,0.780 ,0.777 ,0.955 ,0.614 ,0.939 ]
   classifier_color=['#FF0000','#008000','#0000FF','#FFA500','#800080']
   features_shape=['^','v','o','s','p','o','d','+']
   area = np.pi * 3 ** 2  # 点面积

   plt.rcParams['font.sans-serif'] = ['SimHei']
   plt.rcParams['axes.unicode_minus'] = False
   # matplotlib画图中中文显示会有问题，需要这两行设置默认字体

   plt.xlabel('Accuracy')
   plt.ylabel('Area under curve')
   plt.xlim(xmax=0.88, xmin=0.81)
   plt.ylim(ymax=0.95, ymin=0.8)
   # 画两条（0-9）的坐标轴并设置轴标签x，y
   for i in range(5):
       for j in range(8):
           bg = i*8
           end = bg+8
           plt.scatter(acc1[bg:end], auc1[bg:end], s=area, c=classifier_color[i], alpha=0.8, marker=features_shape[j], label=feature_name[i])
   # plt.scatter(acc1[0:8], auc1[0:8], s=area, c=classifier_color[0], alpha=0.8, marker=features_shape[0],label=feature_name[0])
   # plt.scatter(acc1[8:16], auc1[8:16], s=area, c=classifier_color[1], alpha=0.8, marker=features_shape[1],
   #             label=feature_name[1])
   # plt.scatter(acc1[16:24], auc1[16:24], s=area, c=classifier_color[2], alpha=0.8, marker=features_shape[2],
   #             label=feature_name[2])
   # plt.scatter(acc1[24:32], auc1[24:32], s=area, c=classifier_color[3], alpha=0.8, marker=features_shape[3],
   #             label=feature_name[3])
   # plt.scatter(acc1[32:40], auc1[32:40], s=area, c=classifier_color[4], alpha=0.8, marker=features_shape[4],
   #             label=feature_name[4])
   #plt.legend()
   #plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='类别B')
   #plt.plot([0, 9.5], [9.5, 0], linewidth='0.5', color='#000000')
   #plt.legend()
   #plt.savefig(r'C:\Users\jichao\Desktop\大论文\12345svm.png', dpi=300)
   plt.show()


