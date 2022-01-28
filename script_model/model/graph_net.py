import torch
import torch.nn as nn
import torch.nn.functional as F

class Graph_Net(nn.Module):
    def __init__(self, in_planes):
        super(Graph_Net, self).__init__()

        self.in_planes  = in_planes
        self.out_planes = in_planes//2
        # embedding
        self.query_encoder = nn.Conv2d(in_planes, in_planes//2, kernel_size=1)
        self.key_encoder   = nn.Conv2d(in_planes, in_planes//2, kernel_size=1)
        self.gcn_weight    = nn.Conv2d(in_planes, in_planes//2, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    # cross-modality graph
    def build_graph(self, q_feat, k_feat):
        n, c, h, w = q_feat.shape
        q_feat = q_feat.contiguous().view(n, c, h * w)
        k_feat = k_feat.contiguous().view(n, c, h * w)

        # sparse graph
        graph = torch.bmm(F.normalize(q_feat, dim=1, p=2.).permute(0, 2, 1), F.normalize(k_feat, dim=1, p=2.)).clamp(min=-1, max=1.)
        graph = torch.exp(graph-1).pow(2)
        mask = torch.bernoulli(graph.clamp(0, 1))#.cuda()
        graph = mask * graph

        D = (graph).sum(dim=-1).clamp(min=1e-12).pow(-1).diag_embed()
        return torch.bmm(D, graph)

    def forward(self, x_feat, y_feat):

        # f_feat: n * c * h * w
        # b_feat: n * c * h * w
        q_feat = self.query_encoder(y_feat)
        k_feat = self.key_encoder(x_feat)
        xw     = self.gcn_weight(x_feat)
        
        n, c, h, w = xw.shape
        graph = self.build_graph(q_feat, k_feat)
        # generate final feature
        xw = xw.contiguous().view(n, c, -1).permute(0, 2, 1)
        graph_feat = torch.bmm(graph, xw).permute(0, 2, 1).contiguous().view(n, c, h, w)
        output = self.gamma * graph_feat + q_feat
        return output
