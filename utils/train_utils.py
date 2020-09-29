import itertools


def generate_vmix_search_shells():
    dataset = ['pubmed']
    num_gcn = [1, 2, 3]
    num_vsgc = [2, 4, 6, 8, 12]
    dropout = [0, 0.5, 0.8]
    learn_rate = [0.5, 0.3, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    filename = ['VMIX_search']
    with open('../shells/{}.sh'.format(filename[0]), 'w') as f:
        f.write('#! /bin/bash\n')
        for _ in range(3):
            params = itertools.product(dataset, num_gcn, num_vsgc, dropout, learn_rate, weight_decay, filename)
            for p in params:
                command = 'python train_vmix.py --dataset {} --num_gcn {}  --num_vsgc {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 1\n'.format(p[0], p[1], p[2],
                                                                                              p[3], p[4], p[5], p[6])
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


def generate_vsgc_search_shells():
    # dataset = ['ogbn-arxiv']
    # num_layers = [2, 4, 8, 16, 24]
    # alpha = [1]
    # lambd = [1]
    # dropout = [0, 0.5, 0.8]
    # learn_rate = [0.5, 0.3, 0.1, 0.01]
    # weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    # filename = ['VSGC_search']
    dataset = ['pubmed']
    num_layers = [40]
    alpha = [0.2, 0.4, 0.6, 0.8, 1]
    lambd = [0.4, 0.6, 0.8, 1, 1.2, 1.4]
    dropout = [0.8]
    learn_rate = [0.5]
    weight_decay = [5e-4]
    filename = ['VSGC_search']
    with open('../shells/tmp.sh', 'w') as f:
        f.write('#! /bin/bash\n')
        for _ in range(3):
            params = itertools.product(dataset, num_layers, alpha, lambd, dropout, learn_rate, weight_decay, filename)
            for p in params:
                command = 'python train_vsgc.py --dataset {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 0\n'.format(p[0], p[1], p[2], p[3], p[4],
                                                                                     p[5], p[6], p[7])
                f.write(command)


def generate_vsgc_result_shells():
    with open('../shells/VSGC_result.sh', 'w') as f:
        f.write('#! /bin/bash\n')
        params = []
        #params.append({'dataset': 'cora', 'num_layers': 24, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.8, 'seed': 42, 'cuda': 0, 'learn_rate': 0.3, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'filename': 'VSGC_search', 'train_loss': 0.48666566610336304, 'train_acc': 0.9428571428571428, 'val_loss': 0.8598686456680298, 'val_acc': 0.81, 'test_loss': 0.8203441500663757, 'test_acc': 0.829})
        #params.append({'dataset': 'citeseer', 'num_layers': 16, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.8, 'seed': 42, 'cuda': 1, 'learn_rate': 0.3, 'weight_decay': 0.001, 'num_epochs': 1500, 'patience': 100, 'filename': 'VSGC_search', 'train_loss': 0.760978639125824, 'train_acc': 0.8916666666666667, 'val_loss': 1.2258470058441162, 'val_acc': 0.738, 'test_loss': 1.191381812095642, 'test_acc': 0.727})
        params.append({'dataset': 'pubmed', 'num_layers': 32, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.8, 'seed': 42, 'cuda': 2, 'learn_rate': 0.5, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'filename': 'VSGC_search', 'train_loss': 0.23241811990737915, 'train_acc': 0.9833333333333333, 'val_loss': 0.5322089195251465, 'val_acc': 0.836, 'test_loss': 0.5656287670135498, 'test_acc': 0.818})

        #params.append({'dataset': 'cora', 'num_layers': 24, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.8, 'seed': 42, 'learn_rate': 0.1, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'cuda': 0, 'filename': 'VSGC_search', 'train_loss': 0.4818480908870697, 'train_acc': 0.9428571428571428, 'val_loss': 0.8494647145271301, 'val_acc': 0.806, 'test_loss': 0.8107437491416931, 'test_acc': 0.83})
        #params.append({'dataset': 'citeseer', 'num_layers': 16, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.5, 'seed': 42, 'learn_rate': 0.3, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'cuda': 1, 'filename': 'VSGC_search', 'train_loss': 0.53924161195755, 'train_acc': 0.9333333333333333, 'val_loss': 1.074293851852417, 'val_acc': 0.752, 'test_loss': 1.0391258001327515, 'test_acc': 0.737})
        #params.append({'dataset': 'pubmed', 'num_layers': 32, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.8, 'seed': 42, 'learn_rate': 0.5, 'weight_decay': 5e-05, 'num_epochs': 1500, 'patience': 100, 'cuda': 2, 'filename': 'VSGC_search', 'train_loss': 0.08078866451978683, 'train_acc': 0.9833333333333333, 'val_loss': 0.5012359023094177, 'val_acc': 0.818, 'test_loss': 0.5565839409828186, 'test_acc': 0.81})

        for ps in params:
            for _ in range(100):
                command = 'python train_vsgc.py --dataset {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename VSGC --cuda 3\n'.format(ps['dataset'],
                            ps['num_layers'], ps['alpha'], ps['lambd'], ps['dropout'], ps['learn_rate'],
                                                                                                ps['weight_decay'])
                f.write(command)


if __name__ == '__main__':
    generate_vsgc_result_shells()