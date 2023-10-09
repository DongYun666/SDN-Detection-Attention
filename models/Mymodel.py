from einops import rearrange
import torch
import torch.nn as nn
from models.MLP import MLP

from models.attn import FullAttention, LinearAttentionLayer
from models.encoder import TCEncoder, TCEncoderLayer


class NumEmbedding(nn.Module):
    def __init__(self, dim,num_features) :
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_features, dim))
        self.biases = nn.Parameter(torch.randn(num_features, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases
    

class TC(nn.Module):
    def __init__(self, number_feature_num,category_feature_num, e_layers, num_heads, num_classes,device,dropout = 0.1,d_model = 512):
        super(TC, self).__init__()
        self.device = device
        self.num_embedding = NumEmbedding(dim=d_model,num_features=number_feature_num)
        self.categorical_embeds = nn.ModuleList()
        for i in range(len(category_feature_num)):
            self.categorical_embeds.append(nn.Embedding(category_feature_num[i], d_model))
        
        self.encoder = TCEncoder(
            [
                TCEncoderLayer(
                    LinearAttentionLayer(
                        FullAttention(),
                        d_model= d_model//(2 ** i),
                        num_heads=num_heads
                    ),
                    d_model=d_model//(2 ** i),
                    number_feature_num=number_feature_num,
                    category_feature_num=len(category_feature_num),
                    dropout=dropout
                ) for i in range(e_layers)
            ]
        )
        self.CNN1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm1d(number_feature_num + len(category_feature_num))
        self.act1 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2) 
        self.CNN2 = nn.Conv1d(in_channels=d_model//2, out_channels=d_model//2, kernel_size=3, stride=1, padding=1)
        
        self.norm = nn.LayerNorm(d_model)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Linear(d_model//(2 ** e_layers) * (number_feature_num + len(category_feature_num)), num_classes)
    
    def forward(self,x_number,x_category):
            
            # position embedding
            category = torch.tensor([],device = self.device)
            # 对每一个离散型变量进行embedding
            for i in range(len(self.categorical_embeds)):
                category = torch.cat([category,self.categorical_embeds[i](x_category[:,i].reshape(-1,1))],dim=1)

            x_number = self.num_embedding(x_number)
            
            x = torch.cat([category,x_number],dim=1)
    
            out = self.encoder(x)
    
            out = torch.flatten(out,1)
    
            out = self.head(out)
    
            return out


