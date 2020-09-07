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
    # sgc = [0.766, 0.765, 0.77, 0.765, 0.763, 0.76, 0.755, 0.759, 0.757, 0.751, 0.748, 0.749, 0.753, 0.747, 0.746, 0.744, 0.744, 0.741, 0.734, 0.734, 0.733, 0.732, 0.732, 0.732, 0.733, 0.733, 0.732, 0.731, 0.73, 0.73, 0.73, 0.728, 0.727, 0.73, 0.729, 0.73, 0.73, 0.729, 0.729, 0.729, 0.729, 0.73, 0.729, 0.729, 0.732, 0.732, 0.732, 0.732, 0.733]
    # vsgc1 = [0.746, 0.75, 0.759, 0.758, 0.76, 0.764, 0.766, 0.767, 0.771, 0.771, 0.777, 0.775, 0.777, 0.781, 0.783, 0.777, 0.779, 0.784, 0.786, 0.782, 0.783, 0.783, 0.787, 0.786, 0.788, 0.787, 0.787, 0.789, 0.788, 0.79, 0.788, 0.79, 0.789, 0.79, 0.787, 0.789, 0.789, 0.79, 0.788, 0.79, 0.791, 0.792, 0.792, 0.794, 0.791, 0.789, 0.792, 0.788, 0.789]
    # vsgc05 = [0.747, 0.75, 0.759, 0.759, 0.759, 0.763, 0.766, 0.768, 0.772, 0.775, 0.776, 0.776, 0.777, 0.78, 0.779, 0.778, 0.782, 0.781, 0.782, 0.785, 0.782, 0.787, 0.783, 0.788, 0.787, 0.789, 0.786, 0.788, 0.788, 0.791, 0.788, 0.792, 0.791, 0.791, 0.791, 0.791, 0.79, 0.791, 0.793, 0.793, 0.792, 0.792, 0.788, 0.793, 0.791, 0.789, 0.791, 0.787, 0.788]
    #
    # draw_test_layer('pubmed', sgc, vsgc1, vsgc05)

    sgc = extract_test_accs('../result/train_result/SGC.txt')
    vsgc = extract_test_accs('../result/train_result/VSGC.txt')
    print(sgc)
    print(vsgc[0:49])
    print(vsgc[49:98])
