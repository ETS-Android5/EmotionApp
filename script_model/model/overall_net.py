import torch
from torch import nn
import torch.nn.functional as F
from .baseline import Baseline
from .graph_net import Graph_Net



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class Net(nn.Module):
    def __init__(self, cfg, n_classes):
        super(Net, self).__init__()
        # face and context encoder
        self.f_model = Baseline(cfg)
        self.c_model = Baseline(cfg)
        # cross modality gcn layer
        self.f_graph_module = Graph_Net(self.f_model.in_planes)
        self.c_graph_module = Graph_Net(self.c_model.in_planes)
        # avg pool => generate final features
        self.gap = nn.AdaptiveAvgPool2d(1)
        # batch norm
        self.bottleneck = nn.BatchNorm1d(self.f_model.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        # classification weights
        self.weight  = nn.Parameter(torch.FloatTensor(self.f_model.in_planes, n_classes))
        nn.init.xavier_uniform_(self.weight)

        self.p_m = 0.75
        self.n_m = 1 - self.p_m
        self.s = 20

    def forward(self, face, context):#, label = None):
        
        b = face.shape[0]
        f_feat = self.f_model(face)
        c_feat = self.c_model(context)
        # context & face graph
        f_graph_feat = self.gap(self.f_graph_module(f_feat, c_feat)).view(b, -1)
        c_graph_feat = self.gap(self.c_graph_module(c_feat, f_feat)).view(b, -1)
        graph_feat   = self.bottleneck(torch.cat([f_graph_feat, c_graph_feat], dim=1))
        # predict => sphere mapping

        # ----- 20210727 ----- #
        # output = torch.mm(F.normalize(graph_feat, dim=-1),
        #                   F.normalize(self.weight, dim=0)).clamp(min=-1, max=1.)
        features = F.normalize(graph_feat, dim=-1)  # features before classifier, Size([N, 256])
        output = torch.mm(features,  # the features before classifier, Size([N, 256]), extract it!
                          F.normalize(self.weight, dim=0)).clamp(min=-1, max=1.)
        # -------------------- #
        # if self.training and label is not None:
        #     one_hot = torch.zeros(output.shape, device='cpu')#'cuda')
        #     one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        #     output = one_hot * (output - 0.1) + (1 - one_hot) * output
        #     # boundary loss with orthogonal property 
        #     pos_loss = torch.relu(self.p_m - output[one_hot.to(torch.bool)].flatten())
        #     neg_loss = torch.relu(output[(1 - one_hot).to(torch.bool)].flatten() - self.n_m)
        #     restrict_loss = torch.cat([pos_loss, neg_loss], dim=0).norm(p=2) / b
        #     return output * self.s, self.gap(f_feat).view(b, -1), self.gap(c_feat).view(b, -1), restrict_loss
        # else:
        #     # ----- 20220119 ----- #
        #     # ----- 20210727 ----- #
        #     print("used overall.py line76")
        return output
            # return output, features  # return features before classifier, Size([N, 256])
            # -------------------- #
            # -------------------- #
