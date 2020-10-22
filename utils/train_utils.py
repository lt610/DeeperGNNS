import itertools


def generate_vmix_search_shells():
    dataset = ['pubmed']
    num_inLayer = [1, 2, 3]
    num_vsgc = [2, 4, 6, 8, 12, 16, 24, 32, 40, 48]
    dropout = [0, 0.5, 0.8]
    learn_rate = [0.5, 0.3, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    filename = ['GCN_VSGC_search']
    with open('../shells/{}_{}.sh'.format(filename[0], '_'.join(dataset)), 'w') as f:
        f.write('#! /bin/bash\n')
        for _ in range(5):
            params = itertools.product(dataset, num_inLayer, num_vsgc, dropout, learn_rate, weight_decay, filename)
            for p in params:
                command = 'python train_vmix.py --dataset {} --num_inLayer {}  --num_vsgc {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 2\n'.format(p[0], p[1], p[2],
                                                                                              p[3], p[4], p[5], p[6])
                f.write(command)


def generate_vmix_result_shells():
    with open('../shells/GCN_VSGC_result.txt', 'w') as f:
        # f.write('#! /bin/bash\n')
        params = []

        params.append({'dataset': 'pubmed', 'num_hidden': 64, 'num_inLayer': 1, 'num_vsgc': 6, 'batch_norm': False, 'pair_norm': False, 'residual': False, 'dropout': 0.5, 'dropedge': 0, 'seed': 42, 'learn_rate': 0.3, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'cuda': 3, 'filename': 'VMIX_search', 'train_loss': 0.1729842871427536, 'train_acc': 0.9833333333333333, 'val_loss': 0.5529578924179077, 'val_acc': 0.816, 'test_loss': 0.5784546732902527, 'test_acc': 0.796})
        #params.append({'dataset': 'citeseer', 'num_hidden': 64, 'num_gcn': 1, 'num_vsgc': 12, 'batch_norm': False, 'pair_norm': False, 'residual': False, 'dropout': 0.0, 'dropedge': 0, 'seed': 42, 'learn_rate': 0.5, 'weight_decay': 0.01, 'num_epochs': 1500, 'patience': 100, 'cuda': 2, 'filename': 'VMIX_search', 'train_loss': 0.8454991579055786, 'train_acc': 0.8833333333333333, 'val_loss': 1.2498258352279663, 'val_acc': 0.708, 'test_loss': 1.2216248512268066, 'test_acc': 0.703})
        #params.append({'dataset': 'pubmed', 'num_hidden': 64, 'num_gcn': 1, 'num_vsgc': 6, 'batch_norm': False, 'pair_norm': False, 'residual': False, 'dropout': 0.5, 'dropedge': 0, 'seed': 42, 'learn_rate': 0.3, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'cuda': 3, 'filename': 'VMIX_search', 'train_loss': 0.1729842871427536, 'train_acc': 0.9833333333333333, 'val_loss': 0.5529578924179077, 'val_acc': 0.816, 'test_loss': 0.5784546732902527, 'test_acc': 0.796})

        for ps in params:
            for _ in range(100):
                command = 'python train_vmix.py --dataset {} --num_inLayer {}  --num_vsgc {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename GCN_VSGC_result --cuda 0\n'.format(ps['dataset'],
                            ps['num_inLayer'], ps['num_vsgc'], ps['dropout'], ps['learn_rate'], ps['weight_decay'])
                f.write(command)


def generate_asgc_search_shells():
    dataset = ['cora', 'citeseer', 'pubmed']
    # num_layers = [2, 4, 8, 12, 16]
    num_layers = [24, 32, 40, 48]
    dropout = [0, 0.5, 0.8]
    learn_rate = [0.5, 0.3, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    filename = ['ASGC_search2']
    with open('../shells/{}.sh'.format(filename[0]), 'w') as f:
        f.write('#! /bin/bash\n')
        for _ in range(3):
            params = itertools.product(dataset, num_layers, dropout, learn_rate, weight_decay, filename)
            for p in params:
                command = 'python train_asgc.py --dataset {} --num_layers {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 0\n'.format(p[0], p[1], p[2],
                                                                                              p[3], p[4], p[5])
                f.write(command)


def generate_asgc_result_shells():
    with open('../shells/ASGC_result.sh', 'w') as f:
        f.write('#! /bin/bash\n')
        params = []

        #params.append({'dataset': 'cora', 'num_hidden': 64, 'num_layers': 24, 'dropout': 0.5, 'seed': 42, 'learn_rate': 0.01, 'weight_decay': 0.01, 'num_epochs': 1500, 'patience': 100, 'cuda': 0, 'filename': 'ASGC_search2', 'train_loss': 0.2896374762058258, 'train_acc': 0.9642857142857143, 'val_loss': 0.7122774124145508, 'val_acc': 0.81, 'test_loss': 0.6758201122283936, 'test_acc': 0.841})
        #params.append({'dataset': 'citeseer', 'num_hidden': 64, 'num_layers': 8, 'dropout': 0.5, 'seed': 42, 'learn_rate': 0.01, 'weight_decay': 0.01, 'num_epochs': 1500, 'patience': 100, 'cuda': 0, 'filename': 'ASGC_search', 'train_loss': 0.567916989326477, 'train_acc': 0.975, 'val_loss': 1.1736963987350464, 'val_acc': 0.728, 'test_loss': 1.141464114189148, 'test_acc': 0.725})
        params.append({'dataset': 'pubmed', 'num_hidden': 64, 'num_layers': 24, 'dropout': 0.0, 'seed': 42, 'learn_rate': 0.01, 'weight_decay': 0.01, 'num_epochs': 1500, 'patience': 100, 'cuda': 0, 'filename': 'ASGC_search2', 'train_loss': 0.11917206645011902, 'train_acc': 0.9833333333333333, 'val_loss': 0.5183254480361938, 'val_acc': 0.804, 'test_loss': 0.5337114930152893, 'test_acc': 0.806})

        for ps in params:
            for _ in range(100):
                command = 'python train_asgc.py --dataset {} --num_layers {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename ASGC --cuda 3\n'.format(ps['dataset'],
                            ps['num_layers'], ps['dropout'], ps['learn_rate'], ps['weight_decay'])
                f.write(command)


def generate_agcn_search_shells():
    dataset = ['cora', 'citeseer', 'pubmed']
    # num_layers = [2, 4, 8, 12, 16]
    num_layers = [24, 32, 40, 48]
    dropout = [0, 0.5, 0.8]
    learn_rate = [0.5, 0.3, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    filename = ['AGCN_search2']
    with open('../shells/{}.sh'.format(filename[0]), 'w') as f:
        f.write('#! /bin/bash\n')
        for _ in range(3):
            params = itertools.product(dataset, num_layers, dropout, learn_rate, weight_decay, filename)
            for p in params:
                command = 'python train_agcn.py --dataset {} --num_layers {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 0\n'.format(p[0], p[1], p[2],
                                                                                              p[3], p[4], p[5])
                f.write(command)

def generate_sgc_search_shells():
    dataset = ['pubmed']
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
                command = 'python train_sgc.py --dataset {} --num_layers {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 3\n'.format(p[0], p[1], p[2],
                                                                                              p[3], p[4], p[5])
                f.write(command)


def generate_sgc_result_shells():
    with open('../shells/tmp.sh', 'w') as f:
        f.write('#! /bin/bash\n')
        params = []
        params.append({'dataset': 'cora', 'num_layers': 8, 'pair_norm': False, 'dropout': 0.5, 'seed': 42, 'learn_rate': 0.1, 'weight_decay': 5e-06, 'num_epochs': 1500, 'patience': 100, 'cuda': 1, 'filename': 'SGC_search', 'train_loss': 0.3303910791873932, 'train_acc': 0.95, 'val_loss': 0.7555594444274902, 'val_acc': 0.806, 'test_loss': 0.7171407341957092, 'test_acc': 0.809})
        params.append({'dataset': 'citeseer', 'num_layers': 4, 'pair_norm': False, 'dropout': 0.0, 'seed': 42, 'learn_rate': 0.5, 'weight_decay': 5e-05, 'num_epochs': 1500, 'patience': 100, 'cuda': 1, 'filename': 'SGC_search', 'train_loss': 0.8596817255020142, 'train_acc': 0.9166666666666666, 'val_loss': 1.2720706462860107, 'val_acc': 0.74, 'test_loss': 1.2482657432556152, 'test_acc': 0.712})
        params.append({'dataset': 'pubmed', 'num_layers': 8, 'pair_norm': False, 'dropout': 0.5, 'seed': 42, 'learn_rate': 0.5, 'weight_decay': 5e-06, 'num_epochs': 1500, 'patience': 100, 'cuda': 3, 'filename': 'SGC_search', 'train_loss': 0.22550413012504578, 'train_acc': 0.9666666666666667, 'val_loss': 0.5352100729942322, 'val_acc': 0.828, 'test_loss': 0.5634928345680237, 'test_acc': 0.798})
        for ps in params:
            for _ in range(100):
                command = 'python train_sgc.py --dataset {} --num_layers {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename SGC --cuda 0\n'.format(ps['dataset'],
                            ps['num_layers'], ps['dropout'], ps['learn_rate'], ps['weight_decay'])
                f.write(command)

# 需要修改一下
def generate_vsgc_pre_search_shells():
    dataset = ['cornell', 'texas', 'wisconsin']
    num_layers = [2, 4, 8, 16, 24, 32, 40, 48]
    alpha = [1]
    lambd = [1]
    dropout = [0, 0.5, 0.8]
    learn_rate = [0.5, 0.3, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    filename = ['VSGC_search']
    # dataset = ['ogbn-arxiv']
    # num_layers = [2, 4, 8, 16, 24]
    # alpha = [1]
    # lambd = [1]
    # dropout = [0, 0.5, 0.8]
    # learn_rate = [0.5, 0.3, 0.1, 0.01]
    # weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    # filename = ['VSGC_search']
    # dataset = ['pubmed']
    # num_layers = [40]
    # alpha = [0.2, 0.4, 0.6, 0.8, 1]
    # lambd = [0.4, 0.6, 0.8, 1, 1.2, 1.4]
    # dropout = [0.8]
    # learn_rate = [0.5]
    # weight_decay = [5e-4]
    # filename = ['VSGC_search']
    with open('../shells/{}_{}.sh'.format(filename[0], '_'.join(dataset)), 'w') as f:
        f.write('#! /bin/bash\n')
        for _ in range(5):
            params = itertools.product(dataset, num_layers, alpha, lambd, dropout, learn_rate, weight_decay, filename)
            for p in params:
                command = 'python train_vsgc_pre.py --dataset {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 1\n'.format(p[0], p[1], p[2], p[3], p[4],
                                                                                     p[5], p[6], p[7])
                f.write(command)

def generate_vsgc_pre_search_full_shells():
    # dataset = ['chameleon']
    # dataset = ['cora']
    # num_layers = [2, 4, 8, 16, 24, 32, 40, 48]
    # dataset = ['cornell', 'texas', 'wisconsin']
    dataset = ['wisconsin']
    num_layers = [2, 4, 6, 8, 10]
    alpha = [1]
    lambd = [-0.1, -0.05, -0.01, -0.005, -0.001, 0, 0.001, 0.005, 0.01, 0.05, 0.1]
    dropout = [0, 0.5, 0.8]
    learn_rate = [0.5, 0.3, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    filename = ['VSGC_Pre_search_full']
    with open('../shells/{}_{}.sh'.format(filename[0], '_'.join(dataset)), 'w') as f:
        f.write('#! /bin/bash\n')
        for i in range(10):
            params = itertools.product(dataset, num_layers, alpha, lambd, dropout, learn_rate, weight_decay, filename)
            for p in params:
                split = '../data/splits/{}_split_0.6_0.2_{}.npz'.format(p[0], i)
                command = 'python train_vsgc_pre.py --dataset {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 1 --split {}\n'.format(p[0], p[1], p[2], p[3], p[4],
                                                                                     p[5], p[6], p[7], split)
                f.write(command)


def generate_vsgc_pre_result_shells():
    with open('../shells/VSGC_Pre_result.sh', 'w') as f:
        f.write('#! /bin/bash\n')
        params = []
        params.append({'dataset': 'cora', 'num_layers': 24, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.8, 'seed': 42, 'learn_rate': 0.1, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'cuda': 0, 'filename': 'VSGC_search', 'train_loss': 0.4818480908870697, 'train_acc': 0.9428571428571428, 'val_loss': 0.8494647145271301, 'val_acc': 0.806, 'test_loss': 0.8107437491416931, 'test_acc': 0.83})
        params.append({'dataset': 'citeseer', 'num_layers': 16, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.5, 'seed': 42, 'learn_rate': 0.3, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'cuda': 1, 'filename': 'VSGC_search', 'train_loss': 0.53924161195755, 'train_acc': 0.9333333333333333, 'val_loss': 1.074293851852417, 'val_acc': 0.752, 'test_loss': 1.0391258001327515, 'test_acc': 0.737})
        params.append({'dataset': 'pubmed', 'num_layers': 40, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.8, 'seed': 42, 'learn_rate': 0.5, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'cuda': 0, 'filename': 'VSGC', 'train_loss': 0.22487536072731018, 'train_acc': 0.95, 'val_loss': 0.5185860991477966, 'val_acc': 0.83, 'test_loss': 0.549744725227356, 'test_acc': 0.809})

        for ps in params:
            for _ in range(100):
                command = 'python train_vsgc_pre.py --dataset {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename VSGC_Pre --cuda 2\n'.format(ps['dataset'],
                            ps['num_layers'], ps['alpha'], ps['lambd'], ps['dropout'], ps['learn_rate'],
                                                                                                ps['weight_decay'])
                f.write(command)


def generate_vsgc_search_shells():
    dataset = ['pubmed']
    num_k = [2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    num_layers = [1, 2, 3, 4]
    alpha = [1]
    lambd = [1]
    dropout = [0, 0.5, 0.6, 0.7, 0.8]
    learn_rate = [0.5, 0.3, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    filename = ['VSGC_search']
    with open('../shells/{}_{}.sh'.format(filename[0], '_'.join(dataset)), 'w') as f:
        f.write('#! /bin/bash\n')
        for _ in range(5):
            params = itertools.product(dataset, num_k, num_layers, alpha, lambd, dropout, learn_rate, weight_decay, filename)
            for p in params:
                command = 'python train_vsgc.py --dataset {} --num_k {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 0\n'.format(p[0], p[1], p[2], p[3], p[4],
                                                                                     p[5], p[6], p[7], p[8])
                f.write(command)


def generate_vblockgcn_search_shells():
    dataset = ['pubmed']
    k = [1, 2, 4, 6, 8]
    num_blocks = [2, 3, 4, 6]
    alpha = [1]
    lambd = [1]
    dropout = [0, 0.5, 0.8]
    learn_rate = [0.5, 0.3, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    filename = ['VBlockGCN_search']
    with open('../shells/{}_{}.sh'.format(filename[0], '_'.join(dataset)), 'w') as f:
        f.write('#! /bin/bash\n')
        for _ in range(5):
            params = itertools.product(dataset, k, num_blocks, alpha, lambd, dropout, learn_rate, weight_decay, filename)
            for p in params:
                command = 'python train_block_vgcn.py --dataset {} --k {} --num_blocks {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 3\n'.format(p[0], p[1], p[2], p[3], p[4],
                                                                                     p[5], p[6], p[7], p[8])
                f.write(command)

def generate_vblockgcn_result_shells():
    with open('../shells/VBlockGCN_result.sh', 'w') as f:
        f.write('#! /bin/bash\n')
        params = []
        params.append({'dataset': 'cora', 'num_hidden': 64, 'k': 2, 'num_blocks': 2, 'alpha': 1.0, 'lambd': 1.0, 'residual': False, 'dropout': 0.8, 'seed': 42, 'learn_rate': 0.01, 'weight_decay': 0.001, 'num_epochs': 1500, 'patience': 100, 'cuda': 0, 'filename': 'VBlockGCN_search', 'train_loss': 0.18669097125530243, 'train_acc': 0.9785714285714285, 'val_loss': 0.7133713960647583, 'val_acc': 0.802, 'test_loss': 0.6569525003433228, 'test_acc': 0.839})
        params.append({'dataset': 'citeseer', 'num_hidden': 64, 'k': 6, 'num_blocks': 2, 'alpha': 1.0, 'lambd': 1.0, 'residual': False, 'dropout': 0, 'seed': 42, 'learn_rate': 0.5, 'weight_decay': 0.01, 'num_epochs': 1500, 'patience': 100, 'cuda': 2, 'filename': 'VBlockGCN_search', 'train_loss': 0.5880492925643921, 'train_acc': 0.9, 'val_loss': 1.1017258167266846, 'val_acc': 0.738, 'test_loss': 1.058497667312622, 'test_acc': 0.748})
        params.append({'dataset': 'pubmed', 'num_hidden': 64, 'k': 8, 'num_blocks': 2, 'alpha': 1.0, 'lambd': 1.0, 'residual': False, 'dropout': 0.8, 'seed': 42, 'learn_rate': 0.01, 'weight_decay': 0.01, 'num_epochs': 1500, 'patience': 100, 'cuda': 3, 'filename': 'VBlockGCN_search', 'train_loss': 0.24324862658977509, 'train_acc': 0.9666666666666667, 'val_loss': 0.5691866278648376, 'val_acc': 0.826, 'test_loss': 0.5827363729476929, 'test_acc': 0.808})

        for ps in params:
            for _ in range(100):
                command = 'python train_block_vgcn.py --dataset {} --num_hidden {} --k {} --num_blocks {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename VBlockGCN_result --cuda 0\n'.format(ps['dataset'], ps['num_hidden'], ps['k'], ps['num_blocks'],
                            ps['alpha'], ps['lambd'], ps['dropout'], ps['learn_rate'],ps['weight_decay'])
                f.write(command)


if __name__ == '__main__':
    generate_vsgc_pre_result_shells()