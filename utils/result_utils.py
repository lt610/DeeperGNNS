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
    sgc = [0.7795, 0.8005, 0.7975, 0.805 , 0.808 , 0.807 , 0.805 , 0.799 ,
       0.802 , 0.804 , 0.802 , 0.7955, 0.7965, 0.793 , 0.7915, 0.793 ,
       0.789 , 0.787 , 0.7865, 0.7845, 0.777 , 0.775 , 0.7725, 0.771 ,
       0.769 , 0.766 , 0.762 , 0.759 , 0.756 , 0.753 , 0.7495, 0.745 ,
       0.742 , 0.7325, 0.728 , 0.714 , 0.706 , 0.696 , 0.6855, 0.673 ,
       0.6585, 0.643 , 0.632 , 0.62  , 0.5995, 0.585 , 0.5655, 0.547 ,
       0.528 ]
    vsgc1 = [0.7465, 0.775 , 0.776 , 0.7835, 0.783 , 0.786 , 0.788 , 0.793 ,
       0.792 , 0.794 , 0.798 , 0.799 , 0.798 , 0.799 , 0.8005, 0.8015,
       0.8065, 0.8055, 0.807 , 0.8085, 0.809 , 0.8115, 0.811 , 0.812 ,
       0.811 , 0.8115, 0.811 , 0.8105, 0.812 , 0.8115, 0.812 , 0.813 ,
       0.813 , 0.814 , 0.8135, 0.8145, 0.8145, 0.8135, 0.814 , 0.8155,
       0.8145, 0.813 , 0.814 , 0.8145, 0.8155, 0.816 , 0.8155, 0.8145,
       0.815 ]
    vsgc05 = [0.7475, 0.7815, 0.786 , 0.79  , 0.7895, 0.7955, 0.791 , 0.797 ,
       0.7995, 0.8005, 0.798 , 0.799 , 0.803 , 0.807 , 0.807 , 0.81  ,
       0.8085, 0.811 , 0.813 , 0.8145, 0.816 , 0.815 , 0.816 , 0.817 ,
       0.818 , 0.816 , 0.8135, 0.814 , 0.815 , 0.8135, 0.8135, 0.814 ,
       0.8145, 0.814 , 0.814 , 0.815 , 0.8155, 0.8155, 0.816 , 0.8145,
       0.8145, 0.816 , 0.8155, 0.8165, 0.816 , 0.8155, 0.817 , 0.8165,
       0.82  ]
    draw_test_layer('cora', sgc, vsgc1, vsgc05)

    # vsgc = extract_test_accs('../result/train_result/VSGC.txt')
    # print(vsgc)