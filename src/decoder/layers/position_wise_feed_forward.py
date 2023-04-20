"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn
from models.layers.BayesianLayers import LinearGroupNJ


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1, bayes_compression=False):
        super(PositionwiseFeedForward, self).__init__()
        if bayes_compression:
            self.linear1 = LinearGroupNJ(d_model, hidden, cuda=True)  # TODO look at params
            self.linear2 = LinearGroupNJ(hidden, d_model, cuda=True)
        else:
            self.linear1 = nn.Linear(d_model, hidden)
            self.linear2 = nn.Linear(hidden, d_model)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

        # layers including kl_divergence
        self.kl_list = [self.linear1, self.linear2]

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

    def _kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD
