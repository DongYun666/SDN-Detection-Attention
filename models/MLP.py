from torch import nn

class MLP(nn.Module):
    # define model elements
    def __init__(self, d_model,num_classes,dropout):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(d_model, 2*d_model,bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.hidden2 = nn.Linear(2*d_model, d_model,bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.hidden3 = nn.Linear(d_model, num_classes,bias=True)

    #     self.apply(self._init_weight_)

    # def _init_weight_(self,m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.kaiming_uniform_(m.weight,mode='fan_in', nonlinearity='leaky_relu')
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias,0)


    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        if self.training:
            X = self.dropout1(X)

        X = self.hidden2(X)
        X = self.act2(X)
        if self.training:
            X = self.dropout2(X)

        X = self.hidden3(X)

        return X