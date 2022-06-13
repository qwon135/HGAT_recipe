import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import NodeAttentionLayer, RelationAttentionLayer

class HGAT(nn.Module):
    def __init__(self, user_dim, dim_list, n_hidden_unit, r_hidden_unit, nclass, n_dropout, r_dropout, alpha, nheads, device):
        super(HGAT, self).__init__()
        self.n_hidden_unit = n_hidden_unit
        self.r_hidden_unit = r_hidden_unit
        self.n_dropout = n_dropout
        self.r_dropout = r_dropout
        self.alpha = alpha
        self.nheads = nheads
        self.device = device
        
        self.node_level_attentions = []
        for i in range(len(dim_list)):
            self.node_level_attentions.append([NodeAttentionLayer(user_dim, dim_list[i], self.n_hidden_unit, self.n_dropout , self.alpha, device = self.device, concat=True) for _ in range(nheads)])

        self.W = nn.Parameter(torch.zeros(size=(user_dim, self.n_hidden_unit * nheads))).to(self.device)
        self.relation_level_attention = RelationAttentionLayer(self.n_hidden_unit * nheads, self.r_hidden_unit, self.r_dropout, self.alpha, device = self.device)
        
        self.linear_layer = nn.Linear(self.r_hidden_unit, nclass).to(self.device)
        
    def forward(self, x, x_list, adjs):        
        o_list = []
        for i in range(len(x_list)):
            o_x = torch.cat([att(x, x_list[i], adjs[i]) for att in self.node_level_attentions[i]], dim=1)
            o_list.append(o_x)
        x = torch.mm(x, self.W)
        x = F.dropout(x, self.r_dropout, training=self.training)
        
        x = self.relation_level_attention(x, o_list)
        
        x = self.linear_layer(x)
        
        # return F.log_softmax(x, dim=1)
        return x