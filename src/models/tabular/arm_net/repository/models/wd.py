import torch
from ..models.layers import Embedding, Linear, MLP

class WDModel(torch.nn.Module):
    """
    Model:  Wide and Deep
    Ref:    HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """
    def __init__(self, nfield, nfeat, nemb, mlp_layers, mlp_hid, dropout, noutput):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.linear = Linear(nfeat, noutput)
        self.mlp_ninput = nfield*nemb
        self.mlp = MLP(self.mlp_ninput, mlp_layers, mlp_hid, dropout, noutput)

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x_emb = self.embedding(x)                                       # B*F*E
        y = self.linear(x)+\
            self.mlp(x_emb.view(-1, self.mlp_ninput))        # B
        return y