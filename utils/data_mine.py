import numpy as np
import networkx as nx
import torch as th
from dgl import DGLGraph
from dgl.data import citation_graph as citgrh, CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset


def split_data(data, train_ratio=0.6, val_ratio=0.2, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(data))
    train_set_size = int(len(data) * train_ratio)
    val_set_size = int(len(data) * val_ratio)
    train_indices = shuffled_indices[:train_set_size]
    val_indices = shuffled_indices[train_set_size:(train_set_size + val_set_size)]
    test_indices = shuffled_indices[(train_set_size + val_set_size):]
    return data[train_indices], data[val_indices], data[test_indices]


def stratified_sampling_mask(labels, num_classes, train_ratio=0.6, val_ratio=0.2, random_seed=None):
    """"实现了分层抽样"""
    length = len(labels)
    train_mask, val_mask, test_mask = np.zeros(length), np.zeros(length), np.zeros(length)
    for i in range(num_classes):
        indexes = np.where(labels == i)
        tra, val, tes = split_data(indexes[0], train_ratio, val_ratio, random_seed)
        train_mask[tra], val_mask[val], test_mask[tes] = 1, 1, 1
    return th.BoolTensor(train_mask), th.BoolTensor(val_mask), th.BoolTensor(test_mask)


# def load_data(dataset_name):
#     data = None
#     if dataset_name == 'cora':
#         data = citgrh.load_cora()
#     elif dataset_name == 'citeseer':
#         data = citgrh.load_citeseer()
#     elif dataset_name == 'pubmed':
#         data = citgrh.load_pubmed()
#     return data


def load_data_default(dataset_name):
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        if dataset_name == 'cora':
            dataset = CoraGraphDataset()
        if dataset_name == 'citeseer':
            dataset = CiteseerGraphDataset()
        if dataset_name == 'pubmed':
            dataset = PubmedGraphDataset()
        graph = dataset[0]
        graph = graph.remove_self_loop().add_self_loop()
        print(graph)
        features = graph.ndata['feat']
        labels = graph.ndata['label']
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']
        num_feats = features.shape[1]
        num_classes = int(labels.max().item() + 1)
    else:
        dataset = DglNodePropPredDataset(name=dataset_name)
        splitted_mask = dataset.get_idx_split()
        train_mask, val_mask, test_mask = splitted_mask['train'], splitted_mask['valid'], splitted_mask['test']
        graph, labels = dataset[0]
        features = graph.ndata["feat"]
        num_feats = features.shape[1]
        num_classes = (labels.max() + 1).item()
        # add reverse edges
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)
        #add self-loop
        graph = graph.remove_self_loop().add_self_loop()

    return graph, features, labels, train_mask, val_mask, test_mask, num_feats, num_classes
# def load_data_default(dataset_name):
#     """处理从dgl下载的数据并返回结果"""
#     if dataset_name in ['cora', 'pubmed', 'citeseer']:
#         data = load_data(dataset_name)
#         features = th.FloatTensor(data.features)
#         labels = th.LongTensor(data.labels)
#         train_mask = th.BoolTensor(data.train_mask)
#         val_mask = th.BoolTensor(data.val_mask)
#         test_mask = th.BoolTensor(data.test_mask)
#         num_feats = data.features.shape[1]
#         num_classes = data.num_labels
#         graph = data.graph
#         # add self loop
#         graph.remove_edges_from(nx.selfloop_edges(graph))
#         graph = DGLGraph(graph)
#         graph.add_edges(graph.nodes(), graph.nodes())
#     else:
#         data = DglNodePropPredDataset(name=dataset_name)
#         splitted_mask = data.get_idx_split()
#         train_mask, val_mask, test_mask = splitted_mask['train'], splitted_mask['valid'], splitted_mask['test']
#         graph, labels = data[0]
#         features = graph.ndata["feat"]
#         num_feats = features.shape[1]
#         num_classes = (labels.max() + 1).item()
#         # add reverse edges
#         srcs, dsts = graph.all_edges()
#         graph.add_edges(dsts, srcs)
#         #add self-loop
#         print('Total edges before adding self-loop {}'.format(graph.number_of_edges()))
#         graph = graph.remove_self_loop().add_self_loop()
#         print('Total edges after adding self-loop {}'.format(graph.number_of_edges()))
#
#     return graph, features, labels, train_mask, val_mask, test_mask, num_feats, num_classes


def load_data_mine(dataset_name, train_ratio=0.6, val_ratio=0.2, random_seed=None):
    """dgl返回的数据是按照kipf等固定的设置，无法重新设置训练测试集比例等，这个api可以日后进行扩展"""
    data = load_data(dataset_name)
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    num_feats = data.features.shape[1]
    num_classes = data.num_labels
    train_mask, val_mask, test_mask = stratified_sampling_mask(labels, num_classes, train_ratio, val_ratio, random_seed)
    g = data.graph
    """add self loop"""
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, val_mask, test_mask, num_feats, num_classes


def erase_features(features, val_mask, test_mask, p=0, random_seed=None):
    """PairNorm中的结点特征向量缺失的情形"""
    if p != 0:
        if p == 1:
            features[val_mask] = 0
            features[test_mask] = 0
        else:
            if random_seed:
                np.random.seed(random_seed)
            val_indexes = np.where(val_mask == True)[0]
            test_indexs = np.where(test_mask == True)[0]
            shuffled_val_indices = np.random.permutation(len(val_indexes))
            shuffled_test_indices = np.random.permutation(len(test_indexs))
            val_erase = shuffled_val_indices[int(len(val_indexes) * p)]
            test_erase = shuffled_test_indices[int(len(test_indexs) * p)]
            features[val_erase] = 0
            features[test_erase] = 0


def cut_graph(graph, labels, num_classes):
    """将不同类簇间的边去掉，用于验证过平滑"""
    graph = graph.local_var()
    length1 = len(labels)
    indexes = []
    for i in range(num_classes):
        index = np.where(labels == i)
        t = np.zeros(length1)
        t[index[0]] = 1
        indexes.append(t)
    edges = graph.edges()
    length2 = len(edges[0])
    delete_edges = []
    for i in range(length2):
        u, v = edges[0][i], edges[1][i]
        cla = labels[u]
        if indexes[cla][v] == 0:
            delete_edges.append(i)
    graph.remove_edges(delete_edges)
    return graph


def print_data_info(data):
    print('  NumNodes: {}'.format(data.graph.number_of_nodes()))
    print('  NumEdges: {}'.format(data.graph.number_of_edges()))
    print('  NumFeats: {}'.format(data.features.shape[1]))
    print('  NumClasses: {}'.format(data.num_labels))
    print('  NumTrainingSamples: {}'.format(len(np.nonzero(data.train_mask)[0])))
    print('  NumValidationSamples: {}'.format(len(np.nonzero(data.val_mask)[0])))
    print('  NumTestSamples: {}'.format(len(np.nonzero(data.test_mask)[0])))


def print_graph_info(graph):
    print("number of nodes:{}".format(graph.number_of_nodes()))
    print("number of edges:{}".format(graph.number_of_edges()))



