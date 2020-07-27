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
# 按引用赋值
x = th.Tensor([1, 2, 3])
y = x
print(x)
y[0] = 4
print(x)
