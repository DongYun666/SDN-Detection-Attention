import torch
from torch import nn
from torch.nn import functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    def forward(self,predict,target):
        B, W = predict.shape
        logsoftmax = torch.log(torch.exp(predict) / torch.sum(torch.exp(predict), dim=-1).reshape(B,1))
        loss = torch.mean(-torch.sum(target * logsoftmax,dim=-1))
        return loss







