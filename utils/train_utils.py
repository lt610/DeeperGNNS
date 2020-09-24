import itertools


def generate_sgc_search_shells():
    dataset = ['cora', 'citeseer', 'pubmed', 'ogbn-arxiv']
    num_layers = [2, 4, 8]
    dropout = [0, 0.5, 0.8]
    learn_rate = [0.5, 0.3, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    filename = ['SGC_search']
    with open('../shells/tmp.sh', 'w') as f:
        f.write('#! /bin/bash\n')
        for _ in range(3):
            params = itertools.product(dataset, num_layers, dropout, learn_rate, weight_decay, filename)
            for p in params:
                command = 'python train_vsgc.py --dataset {} --num_layers {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 1\n'.format(p[0], p[1], p[2],
                                                                                              p[3], p[4], p[5])
def generate_vsgc_search_shells():
    dataset = ['ogbn-arxiv']
    num_layers = [2, 4, 8, 16, 24]
    alpha = [1]
    lambd = [1]
    dropout = [0, 0.5, 0.8]
    learn_rate = [0.5, 0.3, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    filename = ['VSGC_search']
    with open('../shells/tmp.sh', 'w') as f:
        f.write('#! /bin/bash\n')
        for _ in range(3):
            params = itertools.product(dataset, num_layers, alpha, lambd, dropout, learn_rate, weight_decay, filename)
            for p in params:
                command = 'python train_vsgc.py --dataset {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 2\n'.format(p[0], p[1], p[2], p[3], p[4],
                                                                                     p[5], p[6], p[7])
                f.write(command)

# def generate_vmix_search_shells():
#     dataset = ['pubmed']
#     num_gcn = [1, 2, 3]
#     num_vsgc = [2, 4, 8, 12, 16]
#     alpha = [1]
#     lambd = [1]
#     dropout = [0, 0.5, 0.8]
#     learn_rate = [0.5, 0.3, 0.1, 0.01]
#     weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
#     filename = ['VSGC_search']
#     with open('../shells/tmp.sh', 'w') as f:
#         f.write('#! /bin/bash\n')
#         for _ in range(3):
#             params = itertools.product(dataset, num_layers, alpha, lambd, dropout, learn_rate, weight_decay, filename)
#             for p in params:
#                 command = 'python train_vsgc.py --dataset {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
#                           '--learn_rate {} --weight_decay {} --filename {} --cuda 2\n'.format(p[0], p[1], p[2], p[3], p[4],
#                                                                                      p[5], p[6], p[7])
#                 f.write(command)

def generate_vsgc_result_shells():
    with open('../shells/tmp.sh', 'w') as f:
        f.write('#! /bin/bash\n')
        params = []
        params.append({'dataset': 'cora', 'num_layers': 16, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.8, 'seed': 42, 'cuda': 0, 'learn_rate': 0.5, 'weight_decay': 5e-05, 'num_epochs': 1500, 'patience': 100, 'filename': 'VSGC_search', 'train_loss': 0.1555197834968567, 'train_acc': 0.9857142857142858, 'val_loss': 0.6824527978897095, 'val_acc': 0.818, 'test_loss': 0.598617434501648, 'test_acc': 0.83})
        params.append({'dataset': 'citeseer', 'num_layers': 16, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.5, 'seed': 42, 'cuda': 1, 'learn_rate': 0.3, 'weight_decay': 0.001, 'num_epochs': 1500, 'patience': 100, 'filename': 'VSGC_search', 'train_loss': 0.7328876852989197, 'train_acc': 0.9166666666666666, 'val_loss': 1.2062867879867554, 'val_acc': 0.76, 'test_loss': 1.169481635093689, 'test_acc': 0.72})
        params.append({'dataset': 'pubmed', 'num_layers': 16, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.5, 'seed': 42, 'cuda': 2, 'learn_rate': 0.5, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'filename': 'VSGC_search', 'train_loss': 0.2705511748790741, 'train_acc': 0.9833333333333333, 'val_loss': 0.5914483666419983, 'val_acc': 0.838, 'test_loss': 0.6018803119659424, 'test_acc': 0.791})
        for ps in params:
            for _ in range(100):
                command = 'python train_vsgc.py --dataset {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename VSGC --cuda 0\n'.format(ps['dataset'],
                            ps['num_layers'], ps['alpha'], ps['lambd'], ps['dropout'], ps['learn_rate'],
                                                                                                ps['weight_decay'])
                f.write(command)

if __name__ == '__main__':
    generate_vsgc_search_shells()