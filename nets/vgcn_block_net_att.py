from layers.vgcn_block_att import VGCNBlock
from layers.mlp_layer import MLPLayer
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import dgl.function as fn
import matplotlib.pyplot as plt


class VGCNBlockNet(nn.Module):
    def __init__(self, num_feats, num_classes, k, num_blocks=2, bias=True, alpha=1, lambd=1,
                 feat_drop=0, attention=False, att_drop=0, best_params=None):
        super(VGCNBlockNet, self).__init__()
        # # 每个Block共享一个W
        # self.mlp = MLPLayer(num_feats, num_classes, bias=True, dropout=feat_drop)
        # self.block1 = VGCNBlock(k, alpha, lambd)
        # self.block2 = VGCNBlock(k, alpha, lambd, attention, att_drop=att_drop)

        # 每个Block的W都不同
        if best_params:
            self.mlp1 = MLPLayer(num_feats, num_classes, bias=True, dropout=best_params['dropout'])
            self.block1 = VGCNBlock(best_params['k'], alpha, lambd)
        else:
            self.mlp1 = MLPLayer(num_feats, num_classes, bias=True, dropout=feat_drop)
            self.block1 = VGCNBlock(k, alpha, lambd)
        self.mlp2 = MLPLayer(num_feats, num_classes, bias=True, dropout=feat_drop)
        self.block2 = VGCNBlock(k, alpha, lambd, attention, att_drop=att_drop)

    def forward(self, graph, features):
        # # 每个Block共享一个W
        # initial_features = self.mlp(graph, features)
        # h = self.block1(graph, initial_features, initial_features)
        # h = self.block2(graph, h, initial_features)

        # 每个Block的W都不同

        initial1 = self.mlp1(graph, features)
        h = self.block1(graph, initial1, initial1)

        initial2 = self.mlp2(graph, features)
        h = self.block2(graph, h, initial2)

        return h

    # def compute_top_rate(self, graph, features, noisy_edges, name):
    #     if noisy_edges is None:
    #         return
    #     g = graph.local_var()
    #     g.ndata['h'] = features
    #     g.apply_edges(fn.u_sub_v('h', 'h', 'dif'))
    #     dif = g.edata.pop('dif')
    #     l2 = th.norm(dif, p=2, dim=1)
    #     att = 1 / (2 * l2 + 1)
    #     _, idxs = att.topk(len(noisy_edges), largest=False, sorted=False)
    #     _, idxs2 = att.topk(len(noisy_edges), largest=True, sorted=False)
    #
    #     plt.hist(att.cpu(), range=[0, 1])
    #     plt.savefig("all_{}.png".format(name))
    #     plt.close()
    #     count = 0
    #     for i in range(len(noisy_edges)):
    #         if idxs[i] in noisy_edges:
    #             count += 1
    #     print("1:{}".format(count / len(noisy_edges)))
    #
    #     count = 0
    #     for i in range(len(noisy_edges)):
    #         if idxs2[i] in noisy_edges:
    #             count += 1
    #     print("2:{}".format(count / len(noisy_edges)))
    #
    #     plt.hist(att[noisy_edges].cpu(), range=[0, 1])
    #     plt.savefig("noisy_{}.png".format(name))
    #     plt.close()
    #     return
