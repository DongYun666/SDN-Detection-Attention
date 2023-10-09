from collections import defaultdict
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# 提前划分好数据集
def process_data(data_path, dataset_name,flag):
    # 加载数据集
    data = np.load(data_path + "/" +flag+ ".npy", allow_pickle=True)
    # 对数据集切分 标签和数据
    value = data[:,:-1].astype(np.float32)
    label = data[:,-1]
    # print(type(value[0,0]))
    # 对数据中的离散型变量进行labelencoder
    # 统计标签数量
    labels_name,labels_num_count = np.unique(label, return_counts=True)

    one_hot_encoder = OneHotEncoder(sparse=False)
    label = one_hot_encoder.fit_transform(label.reshape(-1,1))

    return value, label, labels_name, labels_num_count 

class Dataset(object):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

def get_loader(data_path, batch_size,dataset_name,flag):
    data,labels,labels_name,labels_num_count = process_data(data_path, dataset_name,flag)
    dataset = Dataset(data,labels)
    dataloader = DataLoader(dataset = dataset,batch_size=batch_size,shuffle=True)
    return dataloader,labels_name,labels_num_count




