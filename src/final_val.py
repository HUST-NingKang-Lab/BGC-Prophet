from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

class final_validation:
    def __init__(self, x_true, x_pred, outpath) -> None:
        self.x_true = x_true
        self.x_pred = x_pred
        # self.y_true = y_true
        # self.y_pred = y_pred
        self.outpath = outpath

    def table(self, x_true, x_pred, label):
        # 将概率值转为预测标签
        x_pred = [1 if x >= 0.5 else 0 for x in x_pred]
        # 计算混淆矩阵
        cm = confusion_matrix(x_true, x_pred)
        print(f'confusion-matrix:\n{cm}')
        # 计算每个元素的百分比
        cm_percent = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        tn = cm_percent[0][0]
        fp = cm_percent[0][1]
        fn = cm_percent[1][0]
        tp = cm_percent[1][1]
        print('TP:\t{:.2f}%\tFN:\t{:.2f}%\nFP:\t{:.2f}%\tTN:\t{:.2f}%'.format(tp,fn,fp,tn))
        accuracy = 100*(tp+tn)/(tp+tn+fp+fn)
        recall = 100*tp/(tp+fn)
        precision = 100*tp/(tp+fp)
        F1_Score = 2*precision*recall/(precision+recall)
        print('Accuracy={:.2f}%\tRecall={:.2f}%\tprecision={:.2f}%\tF1-score={:2f}'.format(accuracy,recall,precision,F1_Score))

        # 绘制混淆矩阵
        fig, ax = plt.subplots(1,3,figsize=(20,8))
        # cm
        im0 = ax[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(im0, ax=ax[0])
        ticks = np.arange(2)
        ax[0].set_xticks(ticks)
        ax[0].set_yticks(ticks)
        ax[0].set_xticklabels(['0 (N)', '1 (P)'])
        ax[0].set_yticklabels(['0 (N)', '1 (P)'])
        ax[0].set_xlabel('Predicted label')
        ax[0].set_ylabel('True label')
        ax[0].set_title('Confusion Matrix',fontsize=16,fontweight ="bold")
        thresh = cm.max() / 2
        for i, j in np.ndindex(cm.shape):
            ax[0].text(j, i, format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        # cm_percent
        im1 = ax[1].imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(im1, ax=ax[1])
        ticks = np.arange(2)
        ax[1].set_xticks(ticks)
        ax[1].set_yticks(ticks)
        ax[1].set_xticklabels(['0 (N)', '1 (P)'])
        ax[1].set_yticklabels(['0 (N)', '1 (P)'])
        ax[1].set_xlabel('Predicted label')
        ax[1].set_ylabel('True label')
        ax[1].set_title('Percent Confusion Matrix',fontsize=16,fontweight ="bold")
        thresh = cm_percent.max() / 2
        for i, j in np.ndindex(cm_percent.shape):
            ax[1].text(j, i, format(cm_percent[i, j], '.2f') + '%',
                     horizontalalignment="center",
                     color="white" if cm_percent[i, j] > thresh else "black")
        # table
        cell_text = [['{:.2f}%'.format(accuracy)],['{:.2f}%'.format(recall)],['{:.2f}%'.format(precision)],['{:.2f}%'.format(F1_Score)]]
        row_labels = ['Weighted Accuracy','Weighted Recall','Weighted Precision','Weighted F1-score']
        col_widths = [0.5, 0.5]
        tb = ax[2].table(cellText=cell_text, rowLabels=row_labels, loc='right', cellLoc='center', colWidths=col_widths, rowColours =["palegreen"]*4)
        tb.auto_set_column_width(True)
        tb.set_fontsize(14)  # 设置字体大小为14
        ax[2].axis('off')
        ax[2].set_title('Weighted Evaluation',fontsize=16,fontweight ="bold")
        # 保存图片
        file_path = self.outpath + f'_{label}_cm.png'
        fig.savefig(file_path, dpi=600, bbox_inches='tight')
    
    #def F1_score(self,x_true, x_pred):# 按照个数来计算
    #    x_pred = [int(item>=1) for  item in x_pred]
    #    f1 = f1_score(x_true.astype('int'),x_pred)
    #    print("F1-Score:{:.4f}".format(f1))

    def plot_step(self, x_true, x_pred, x):
        # x_true是真实标签，x_pred是模型输出的概率值
        # 计算TPR和FPR
        fpr, tpr, thresholds = roc_curve(x_true.astype('int'), x_pred)
        # 计算AUC
        roc_auc = auc(fpr, tpr)
        # 绘制ROC曲线
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{x} ROC')
        plt.legend(loc="lower right")
        file_path = self.outpath + f'_{x}_ROC.png'
        plt.savefig(file_path, dpi=300)

    def plot_two(self):
        # x_true是真实标签，x_pred是模型输出的概率值
        # 计算TPR和FPR
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr['gene'], tpr['gene'], _ = roc_curve(self.x_true.astype('int'), self.x_pred)
        fpr['gene'], tpr['gene'], _ = roc_curve(self.x_true.astype('int'), self.x_pred)
        # 计算AUC
        # roc_auc['bgc'] = auc(fpr['bgc'], tpr['bgc'])
        roc_auc['gene'] = auc(fpr['gene'], tpr['gene'])
        # 绘制ROC曲线
        plt.figure(figsize=(8, 8))
        # plt.plot(fpr['bgc'], tpr['bgc'], color='blue', lw=2, label=' BGC ROC curve (area = %0.2f)' % roc_auc['bgc'])
        plt.plot(fpr['gene'], tpr['gene'], color='red', lw=2, label=' Gene ROC curve (area = %0.2f)' % roc_auc['gene'])
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('BGC and Gene ROC')
        plt.legend(loc="lower right")
        file_path = self.outpath + f'_Gene_ROC.png'
        plt.savefig(file_path, dpi=300)

    def result(self):
        # print('Final validation of BGC:')
        # self.plot_step(self.x_true,self.x_pred,'BGC')
        # self.table(self.x_true,self.x_pred,'BGC')
        print('Final validation of Gene:')
        self.plot_step(self.x_true,self.x_pred,'Gene')
        self.table(self.x_true,self.x_pred,'Gene')
        # self.plot_two()

