import itertools
import matplotlib.pyplot as plt
from dgl.data import citation_graph as citegrh
from sklearn.manifold import Isomap
from sklearn.grid_search import GridSearchCV
from dgl import DGLGraph
import numpy as np
import torch as th
import torch.nn.functional as F
import dgl.function as fn
from utils.data_geom import load_data_from_file
from utils.data_mine import print_graph_info
import os

"""测试按引用赋值"""
# x = th.Tensor([1, 2, 3])
# y = x
# print(x)
# y[0] = 4
# print(x)


"""测试geom中引入的data相关的api"""
"""citation数据集即使加了自循环，也不应该有这么多边啊，可能和作者disconnected那部分有关，但是没看懂"""
"""添加了自循环后，chameleon的边数少了50，不知道是什么原因"""
# g, features, labels, train_mask, val_mask, test_mask, num_feats,\
#                 num_classes = load_data_from_file('chameleon', None, 0.6, 0.2)
# print_graph_info(g)

"""测试命令行"""
# cmds = []
# cmds.append('python hello.py --name A')
# cmds.append('python hello.py --name B')
# for cmd in cmds:
#     os.system(cmd)

"""测试读命令文件"""
with open('shells/test.txt', 'r') as f:
    cmds = f.readlines()
    for cmd in cmds:
        print(cmd)