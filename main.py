import os
import argparse
import time

from torch.backends import cudnn
from utils.utils import *

from Experment import Experiment

def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    for i in range(config.num_Experiment):
        print("============= 进行第{}次实验=============".format(i+1))
        experiment= Experiment(vars(config))
        if config.mode == 'train':
            start_time = time.time()
            experiment.train()
            print("训练时间：{}".format(time.time() - start_time))
        else:
            experiment.test()
    return experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 模型部分参数

    parser.add_argument('--number_feature_num', type=int, default=59,help="连续型变量特征个数")
    parser.add_argument('--category_feature_num', type=int, default=[2,2,2,2,2,2,2],help="离散型变量种类数")
    # parser.add_argument('--number_feature_num', type=int, default=8,help="连续型变量特征个数")
    # parser.add_argument('--category_feature_num', type=list, default=[3,2,3],help="离散型变量种类数")
    # parser.add_argument('--category_feature_num', type=int, default=3,help="离散型变量种类数")

    parser.add_argument('--e_layers', type=int, default=3,help="encoder 的层数")
    parser.add_argument('--num_heads', type=int, default=8,help="划分的头的个数")

    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--padding', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    # parser.add_argument('--wavelet_method', type=str, default='sym4')

    parser.add_argument('--trainable', type=bool, default=True)


    # 实验部分参数
    parser.add_argument('--device', type=str, default='cuda:0' ,help="device设置")
    parser.add_argument('--num_Experiment', type=int, default=1 ,help="实验次数")
    parser.add_argument('--lr', type=float, default=3e-4) # 3e-4 的时候出现了一次0.95
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20,help="早停机制中的patience参数")
    parser.add_argument('--random_seed', type=int, default=226,help="设置随机种子")
    parser.add_argument('--mode', type=str, default='train',help="选择模式")
    # parser.add_argument('--mode', type=str, default='test',help="选择模式")

    # 数据集部分
    parser.add_argument('--data_path',type=str,default='processdata/CIC_IDS_2017')
    parser.add_argument('--dataset_name',type=str,default='')

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--win_size', type=int, default=100,help="窗口大小")


    config = parser.parse_args()
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)

