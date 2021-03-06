import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch as th


def get_noisy_edges(graph, labels):
    graph = graph.local_var()
    edges = graph.edges()
    ne = len(edges[0])
    nosiy_edges = []
    for i in range(ne):
        u, v = edges[0][i], edges[1][i]
        if labels[u] != labels[v]:
            nosiy_edges.append(i)

    nosiy_edges = th.LongTensor(nosiy_edges)
    return nosiy_edges



def draw_loss_acc(train_loss, val_loss, test_loss, train_acc, val_acc, test_acc):
    x = np.arange(0, 1500, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, train_loss, color='green', linestyle='-', label="train_loss")
    ax1.plot(x, val_loss, color='orange', linestyle='-', label="val_loss")
    # ax1.plot(x, test_loss, color='blueviolet', linestyle=':', label="test_loss")
    ax1.legend(loc=0)
    ax1.set_ylabel('Loss')
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, train_acc, color='red', linestyle='-', label="train_acc")
    ax2.plot(x, val_acc, color='blue', linestyle='-', label="val_acc")
    # ax2.plot(x, test_acc, color='blueviolet', linestyle='-', label="test_acc")
    ax2.legend(loc=0)
    ax2.set_xlim([0, 1500])
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    plt.show()


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


def extract_search_result(filename, times=5, topk=4, all=True):
    result = extract_result(filename)
    length = len(result)
    print(length)
    gap = int(length / times)
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

    # 循环输出前k个最大的test_acc值
    test_accs2_mean = np.mean(test_accs2, 0)
    for i in range(topk):
        print('top {}'.format(i+1))
        idx2 = np.where(test_accs2_mean == np.max(test_accs2_mean))
        print('mean_test:{}'.format(np.mean(test_accs2[..., idx2[0][0]])))
        if all:
            print(result2[..., idx2[0][0]])
        else:
            print(result2[0, idx2[0][0]])
        test_accs2_mean[idx2] = 0

    # print('test_acc_max')
    # test_accs2_mean = np.mean(test_accs2, 0)
    # idx2 = np.where(test_accs2_mean == np.max(test_accs2_mean))
    # print('mean_test:{}'.format(np.mean(test_accs2[..., idx2[0][0]])))
    # print(result2[..., idx2[0][0]])



    # print('test_acc_second')
    # test_accs2_mean[idx2] = 0
    # idx2 = np.where(test_accs2_mean == np.max(test_accs2_mean))
    # print('mean_test:{}'.format(np.mean(test_accs2[..., idx2[0][0]])))
    # print(result2[..., idx2[0][0]])
    #
    # print('test_acc_third')
    # test_accs2_mean[idx2] = 0
    # idx2 = np.where(test_accs2_mean == np.max(test_accs2_mean))
    # print('mean_test:{}'.format(np.mean(test_accs2[..., idx2[0][0]])))
    # print(result2[..., idx2[0][0]])
    #
    # print('test_acc_fourth')
    # test_accs2_mean[idx2] = 0
    # idx2 = np.where(test_accs2_mean == np.max(test_accs2_mean))
    # print('mean_test:{}'.format(np.mean(test_accs2[..., idx2[0][0]])))
    # print(result2[..., idx2[0][0]])

def extract_final_result(filename):
    result = extract_result(filename)
    print('{}条数据'.format(len(result)))
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
    print(result[0])


def extract_dropedge_result(filename, times):
    result = extract_result(filename)
    length = len(result)
    print(length)
    gap = int(length / times)
    test_accs = []
    for r in result:
        test_accs.append(r['test_acc'])

    test_accs2 = np.array(test_accs)
    test_accs2.resize([times, gap])
    test_accs2_mean = np.mean(test_accs2, 0)
    print(test_accs2_mean * 100)


def check_missing_cmd(sh_file, out_file):
    shs = []
    with open(sh_file, 'r') as f:
        shs = f.readlines()
        shs.pop(0)
    n = len(shs)
    ids = {}
    for i in range(n):
        ids[i] = 0
    result = extract_result(out_file)
    for r in result:
        ids[r['id']] = 1
    count = 0
    with open("../shells/missing_{}".format(sh_file.split('/')[-1]), 'w') as f:
        for i in range(n):
            if ids[i] == 0:
                count += 1
                print('missing:{}'.format(shs[i]))
                f.write('{}'.format(shs[i]))
    print("一共缺失了{}条数据".format(count))


def insert_missing_out(src_file, des_file):
    result1 = extract_result(src_file)
    result2 = list(extract_result(des_file))

    for r in result1:
        id = r['id']
        result2.insert(id, r)
    with open("../result/train_result/repair_{}".format(des_file.split('/')[-1]), "w") as f:
        for r in result2:
            f.write('{}, '.format(r))


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

    filename = "VBlockGCN_att_search_share_citeseer"

    # extract_dropedge_result('../result/train_result/VBlockGCN_drop_unimportant_cora.txt', 10)

    extract_search_result('../result/train_result/des_result/{}.txt'.format(filename), 3, topk=6, all=True)

    # extract_search_result('../result/train_result/repair_{}.txt'.format(filename), 3)

    # extract_final_result('../result/train_result/final_result/VBlockGCN_att_result1_2_cora.txt')

    # check_missing_cmd("../shells/10.192.9.122/{}.sh".format(filename),
    #                   "../result/train_result/des_result/{}.txt".format(filename))

    # check_missing_cmd("../shells/aws/{}.sh".format(filename),
    #                   "../result/train_result/des_result/{}.txt".format(filename))

    # insert_missing_out('../result/train_result/{}.txt'.format(filename),
    #                    '../result/train_result/des_result/{}.txt'.format(filename))

    # r = extract_result("../result/train_result/{}.txt".format(filename))
    # print(len(r))
