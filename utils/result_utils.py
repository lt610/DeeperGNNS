import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def draw_part_graph(graph, nodes=None):
    if nodes is not None:
        g = graph.subgraph(nodes)
    else:
        g = graph
    pos = nx.kamada_kawai_layout(g)
    nx.draw(g, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.show()


def extract_test_accs(filename):
    results = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line != '':
            words = line.split()
            result = float(words[-1])
            results.append(result)
            line = f.readline()
    return results


def extract_result(filename):
    result = None
    with open(filename, 'r') as f:
        r = f.read()
    result = eval(r)
    return result

def extract_search_result(filename):
    result = extract_result(filename)
    length = len(result)
    print(length)
    times = 10
    gap = int(length/times)
    print(gap)
    val_accs = []
    test_accs = []
    for r in result:
        val_accs.append(r['val_acc'])
        test_accs.append(r['test_acc'])
    result2 = np.array(result)
    result2.resize([times, gap])

    val_accs2 = np.array(val_accs)
    val_accs2.resize([times, gap])

    test_accs2 = np.array(test_accs)
    test_accs2.resize([times, gap])

    print('val_acc_max')
    val_accs2_mean = np.mean(val_accs2, 0)
    idx1 = np.where(val_accs2_mean == np.max(val_accs2_mean))
    print('mean_test:{}'.format(np.mean(test_accs2[..., idx1[0][0]])))
    print(result2[..., idx1[0][0]])

    print('test_acc_max')
    test_accs2_mean = np.mean(test_accs2, 0)
    idx2 = np.where(test_accs2_mean == np.max(test_accs2_mean))
    print('mean_test:{}'.format(np.mean(test_accs2[..., idx2[0][0]])))
    print(result2[..., idx2[0][0]])

def extract_final_result(filename):
    result = extract_result(filename)
    print(len(result))
    val_accs = []
    test_accs = []
    for r in result:
        val_accs.append(r['val_acc'])
        test_accs.append(r['test_acc'])
    test_accs2 = np.array(test_accs)
    mean = np.mean(test_accs2)
    std = np.std(test_accs2)
    print('mean:{}'.format(mean))
    print('std:{}'.format(std))

def draw_test_layer(title, sgc_test, vsgc1_test, vsgc05_test):
    x = np.arange(2, 51)
    plt.title(title)
    plt.xlabel('Number of Layers')
    plt.ylabel('Test Accuracy')
    plt.xticks(x)
    plt.xlim(1.5, 50.5)
    marker = ''
    ms = 4
    plt.plot(x, sgc_test, color='green', linestyle='-', marker=marker, ms=ms, label='sgc')
    plt.plot(x, vsgc1_test, color='orange', linestyle='-', marker=marker, ms=ms, label='vsgc(alpha=1)')
    plt.plot(x, vsgc05_test, color='blueviolet', linestyle='-', marker=marker, ms=ms, label='vsgc(alpha=0.5)')
    plt.legend()
    plt.show()


def draw(title, method, train_accs0, train_accs1, test_accs0, test_accs1):
    # x = np.arange(1, 33)
    x = np.arange(1, 17)
    plt.title(title)
    plt.xlabel('Number of Layers')
    plt.ylabel('Accuracy')
    plt.xticks(x)
    # plt.xlim(0.5, 33.5)
    plt.xlim(0.5, 16.5)
    # plt.ylim(0, 1)
    marker = 'o'
    ms = 4
    plt.plot(x, train_accs0, color='green', linestyle='--', marker=marker, ms=ms, label='Train')
    plt.plot(x, train_accs1, color='orange', linestyle='-', marker=marker, ms=ms, label='Train({})'.format(method))
    plt.plot(x, test_accs0, color='red', linestyle='--', marker=marker, ms=ms, label='Test')
    plt.plot(x, test_accs1, color='blueviolet', linestyle='-', marker=marker, ms=ms, label='Test({})'.format(method))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # sgc = [0.652, 0.651, 0.669, 0.672, 0.67, 0.673, 0.672, 0.677, 0.681, 0.678, 0.675, 0.671, 0.671, 0.678, 0.68, 0.681, 0.683, 0.683, 0.683, 0.68, 0.683, 0.681, 0.682, 0.681, 0.68, 0.682, 0.683, 0.686, 0.686, 0.684, 0.685, 0.685, 0.686, 0.687, 0.686, 0.685, 0.688, 0.688, 0.686, 0.686, 0.685, 0.685, 0.689, 0.688, 0.69, 0.692, 0.692, 0.69, 0.691]
    # vsgc1 = [0.629, 0.631, 0.642, 0.637, 0.644, 0.652, 0.654, 0.654, 0.65, 0.652, 0.649, 0.651, 0.653, 0.658, 0.657, 0.668, 0.669, 0.669, 0.668, 0.674, 0.674, 0.674, 0.68, 0.684, 0.682, 0.686, 0.691, 0.691, 0.69, 0.695, 0.692, 0.695, 0.699, 0.697, 0.7, 0.695, 0.697, 0.698, 0.696, 0.696, 0.695, 0.697, 0.695, 0.694, 0.693, 0.694, 0.691, 0.695, 0.692]
    # vsgc05 = [0.631, 0.631, 0.641, 0.636, 0.644, 0.653, 0.656, 0.655, 0.649, 0.652, 0.65, 0.653, 0.654, 0.657, 0.66, 0.667, 0.665, 0.671, 0.671, 0.671, 0.674, 0.679, 0.679, 0.682, 0.68, 0.689, 0.688, 0.688, 0.695, 0.691, 0.694, 0.692, 0.699, 0.698, 0.695, 0.694, 0.698, 0.698, 0.696, 0.693, 0.693, 0.693, 0.697, 0.693, 0.697, 0.691, 0.693, 0.687, 0.694]
    #
    # draw_test_layer('citeseer', sgc, vsgc1, vsgc05)

    # sgc = extract_test_accs('../result/train_result/SGC.txt')
    # vsgc = extract_test_accs('../result/train_result/VSGC.txt')
    # print(sgc)
    # print(vsgc[0:7])
    # print(vsgc[7:14])

    # result = extract_result('../result/train_result/VSGC_search_ogbn-arxiv.txt')
    # print(len(result))
    # layers = []
    # val_accs = []
    # test_accs = []
    # for r in result:
    #     layers.append(r['num_layers'])
    #     val_accs.append(r['val_acc'])
    #     test_accs.append(r['test_acc'])
    # i1 = val_accs.index(max(val_accs))
    # i2 = test_accs.index(max(test_accs))
    # print(result[i1])
    # print(result[i2])

    extract_search_result('../result/train_result/VSGC_search_pubmed.txt')

    # extract_final_result('../result/train_result/ASGC_pubmed.txt')

