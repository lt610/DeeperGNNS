import itertools

def generate_vsgc_search_shells():
    dataset = ['cora']
    num_layers = [2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    alpha = [1]
    lambd = [1]
    dropout = [0, 0.5, 0.8]
    learn_rate = [0.2, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    filename = ['VSGC_search']
    params = itertools.product(dataset, num_layers, alpha, lambd, dropout, learn_rate, weight_decay, filename)
    with open('../shells/tmp.txt', 'w') as f:
        for _ in range(3):
            for p in params:
                command = 'python train_vsgc.py --dataset {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {}\n'.format(p[0], p[1], p[2], p[3], p[4],
                                                                                     p[5], p[6], p[7])
                f.write(command)


if __name__ == '__main__':
    generate_vsgc_search_shells()