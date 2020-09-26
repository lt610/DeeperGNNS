import itertools


def generate_asgc_search_shells():
    dataset = ['cora', 'citeseer', 'pubmed']
    num_layers = [2, 4, 8, 12, 16]
    dropout = [0, 0.5, 0.8]
    learn_rate = [0.5, 0.3, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    filename = ['ASGC_search']
    with open('../shells/{}.sh'.format(filename[0]), 'w') as f:
        f.write('#! /bin/bash\n')
        for _ in range(3):
            params = itertools.product(dataset, num_layers, dropout, learn_rate, weight_decay, filename)
            for p in params:
                command = 'python train_asgc.py --dataset {} --num_layers {} --dropout {} ' \
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
    with open('../shells/tmp.sh', 'w') as f:
        f.write('#! /bin/bash\n')
        params = []
        params.append({'dataset': 'cora', 'num_layers': 32, 'alpha': 0.8, 'lambd': 0.6, 'dropout': 0.5, 'seed': 42, 'learn_rate': 0.1, 'weight_decay': 5e-05, 'num_epochs': 1500, 'patience': 100, 'cuda': 0, 'filename': 'VSGC_search', 'train_loss': 0.1379988193511963, 'train_acc': 0.9928571428571429, 'val_loss': 0.6923893690109253, 'val_acc': 0.806, 'test_loss': 0.634097695350647, 'test_acc': 0.814})
        params.append({'dataset': 'citeseer', 'num_layers': 16, 'alpha': 1.0, 'lambd': 1.4, 'dropout': 0.5, 'seed': 42, 'learn_rate': 0.3, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'cuda': 0, 'filename': 'VSGC_search', 'train_loss': 0.5406072735786438, 'train_acc': 0.9083333333333333, 'val_loss': 1.074610710144043, 'val_acc': 0.748, 'test_loss': 1.0375072956085205, 'test_acc': 0.736})
        params.append({'dataset': 'pubmed', 'num_layers': 40, 'alpha': 1.0, 'lambd': 0.8, 'dropout': 0.8, 'seed': 42, 'learn_rate': 0.5, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'cuda': 0, 'filename': 'VSGC_search', 'train_loss': 0.22446662187576294, 'train_acc': 0.95, 'val_loss': 0.5114421844482422, 'val_acc': 0.832, 'test_loss': 0.5330830812454224, 'test_acc': 0.802})
        for ps in params:
            for _ in range(100):
                command = 'python train_vsgc.py --dataset {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename VSGC_AB --cuda 0\n'.format(ps['dataset'],
                            ps['num_layers'], ps['alpha'], ps['lambd'], ps['dropout'], ps['learn_rate'],
                                                                                                ps['weight_decay'])
                f.write(command)


if __name__ == '__main__':
    generate_asgc_search_shells()