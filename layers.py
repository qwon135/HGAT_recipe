import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeAttentionLayer(nn.Module):
    def __init__(self, t_in_features, o_in_features, out_features, nd_dropout, alpha, device, concat=True):
        super(NodeAttentionLayer, self).__init__()
        self.nd_dropout = nd_dropout
        self.t_in_features = t_in_features
        self.o_in_features = o_in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.device = device

        self.W_t = nn.Parameter(torch.zeros(size=(t_in_features, out_features))).to(self.device)
        self.W_o = nn.Parameter(torch.zeros(size=(o_in_features, out_features))).to(self.device)
        nn.init.xavier_uniform_(self.W_t.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_o.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1))).to(self.device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha).to(self.device)

    def forward(self, t_input, o_input, adj):
        h_t = torch.mm(t_input, self.W_t)
        h_o = torch.mm(o_input, self.W_o)
        N_t = h_t.size()[0]
        N_o = h_o.size()[0]        
        
        a_input = torch.cat([h_t.repeat(1, N_o).view(N_t * N_o, -1), h_o.repeat(N_t, 1)], dim=1).view(N_t, N_o, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)).to(self.device)

        zero_vec = -9e15*torch.ones_like(e).to(self.device)
        attention = torch.where(adj > 0, e, zero_vec).to(self.device)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.nd_dropout, training=self.training)
        h_prime = torch.matmul(attention, h_o)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.t_in_features) + ' -> ' + str(self.out_features) + ')'

class RelationAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, se_dropout, alpha, device):
        super(RelationAttentionLayer, self).__init__()
        self.se_dropout = se_dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.device = device

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features))).to(self.device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, out_features))).to(self.device)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.s = nn.Parameter(torch.zeros(size=(2*out_features, 1))).to(self.device)
        nn.init.xavier_uniform_(self.s.data, gain=1.414)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, t_input, o_list):
        N = t_input.shape[0]
        h_t = torch.mm(t_input, self.W)

        s_input = torch.cat([h_t,h_t],dim=1)
        for h_o in o_list:
            h_o = torch.mm(h_o, self.W) #+ self.b

            temp = torch.cat([h_t,h_o],dim=1)
            s_input = torch.cat([s_input,temp],dim=1)

        s_input = s_input.view((len(o_list)+1) *N, 2*self.out_features)
        e = self.leakyrelu(torch.mm(s_input, self.s).view(N, -1))
        schema_attentions = F.softmax(e, dim=1)  

        schema_attentions = schema_attentions.unsqueeze(dim=1)

        embed_list = []
        embed_list.append(h_t)
        for h_o in o_list:
            h_o = torch.mm(h_o, self.W)
            embed_list.append(h_o)
        h_embedding = torch.cat(embed_list,dim = 1).view(N, -1 ,self.out_features)        
        h_embedding = torch.matmul(schema_attentions, h_embedding).squeeze()
 
        return h_embedding

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'