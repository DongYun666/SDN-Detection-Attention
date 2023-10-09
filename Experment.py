import os
import time

import numpy as np
import torch
from torch import nn

from data_factory.data_loader import get_loader
from models.Mymodel import TC

import warnings
from sklearn.metrics import confusion_matrix


from models.loss import CrossEntropyLoss

warnings.filterwarnings("ignore")


# 动态调整学习率
def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.9 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))

# def adjust_learning_rate(optimizer, epoch, lr_):
#     lr = lr_ * 0.9
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr
#     print("Updating learning rate to {}".format(lr))

# 早停机制 可能需要修改  如果val_loss
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.var_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, path, num_experiment):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, num_experiment)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, num_experiment)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, num_experiment):
        if self.verbose:
            print(f'====Validation loss decreased ({self.var_loss_min:.6f} --> {val_loss:.6f}). Saving model====')
        torch.save(model.state_dict(),os.path.join(path, str(self.dataset)[:-6] + '_checkpoint_' + str(num_experiment) + '_.pth'))
        self.var_loss_min = val_loss


# 实验主程序
class Experiment(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Experiment.DEFAULTS, **config)
        self.same_seeds(seed=self.random_seed)

        self.data,self.classes,self.labels_num_count = get_loader(data_path=self.data_path, batch_size=self.batch_size, dataset_name= self.dataset_name, flag=self.mode)
        
        self.category_feature_length = len(self.category_feature_num)

        self.classes_num = len(self.classes)

        self.device = torch.device(self.device if self.device is not None else "cpu")

        self.Multi_Class_criterion = nn.CrossEntropyLoss()

        self.Binary_criterion = nn.BCELoss()

        self.my_loss_criterion = CrossEntropyLoss()

        self.build_model()


    def build_model(self):
        # d_model 可能需要更改 

        self.model = TC(
            number_feature_num = self.number_feature_num,         # 连续型变量特征个数
            category_feature_num = self.category_feature_num,     # 离散型变量种类 个数
            e_layers = self.e_layers,
            num_heads = self.num_heads,
            num_classes = self.classes_num,
            device = self.device,
            dropout = self.dropout,
            d_model = self.d_model
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-2,eps=1e-8)
        if torch.cuda.is_available():
            self.model.to(self.device)
        

    def same_seeds(self, seed):
        torch.manual_seed(seed)  # 固定随机种子（CPU）
        if torch.cuda.is_available():  # 固定随机种子（GPU)
            torch.cuda.manual_seed(seed)  # 为当前GPU设置
            torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
        np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
        torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
        torch.backends.cudnn.deterministic = True


    def train(self):
        print('====================Train model======================')
        time_now = time.time()
        path = self.model_save_path

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, dataset_name=self.dataset_name)
        train_steps = len(self.data)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss_list = []
            epoch_time = time.time()
            self.model.train()

            for i, (input_data, label) in enumerate(self.data):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.to(self.device)
                label = label.float().to(self.device)
                output = self.model(input[:,:-self.category_feature_length].float(),input[:,-self.category_feature_length:].long())
                
                loss = self.my_loss_criterion(output,label)
                loss_list.append(loss)

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = torch.mean(torch.tensor(loss_list))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps,train_loss))

            early_stopping(train_loss, self.model, path, self.num_Experiment)
            if early_stopping.early_stop:
               print("Early stopping")
               break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path),str(self.dataset_name)[:-5] + '_checkpoint_' + str(self.num_Experiment) + '_.pth')
            )
        )
        self.model.eval()
        print("=====================Test model============================")
        labels = []
        outputs = []
        for i, (input_data, label) in enumerate(self.data):

            input = input_data.to(self.device)
            output = self.model(input[:,:self.category_feature_length].float(),input[:,self.category_feature_length:].long())
            
            output = output.max(dim=-1)[1]
            label = label.max(dim=-1)[1]

            outputs.extend(output.detach().cpu().numpy().reshape(-1))
            labels.extend(label.detach().cpu().numpy().reshape(-1))

        from sklearn.metrics import classification_report
        report = classification_report(labels, outputs,digits=8)
        print("测试结果：")
        print(report)

        cm = confusion_matrix(y_true=labels, y_pred=outputs, labels=[i for i in range(len(self.classes))])
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[c[3:] for c in self.classes])
        # disp.plot()
        # plt.show()
        # print(disp.confusion_matrix)
        print("混淆矩阵：")
        print(cm)
