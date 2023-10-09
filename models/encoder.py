import torch.nn as nn


class TCEncoderLayer(nn.Module):
    def __init__(self,attention,d_model,number_feature_num,category_feature_num,dropout):
        super(TCEncoderLayer, self).__init__()
        self.attention = attention

        self.norm = nn.BatchNorm1d(number_feature_num + category_feature_num)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2) 

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

        self.dropout = nn.Dropout(p = dropout)
        self.activation = nn.ReLU()

    def forward(self,x):
        x = self.attention(x) + x

        if self.training:
            x = self.dropout(x)

        x = self.norm(x)

        x = self.feedforward(x) + x

        x = self.norm(x)

        x = self.activation(x)

        x = self.maxpool(x)

        return x


class TCEncoder(nn.Module):
    def __init__(self, stage_layers):
        super(TCEncoder, self).__init__()
        self.attn_layers = nn.Sequential(*stage_layers)

    def forward(self, x):
        x = self.attn_layers(x)
        return x