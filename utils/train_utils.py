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
                          '--learn_rate {} --weight_decay {} --filename ASGC_result --cuda 3\n'.format(ps['dataset'],
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
                          '--learn_rate {} --weight_decay {} --filename SGC_result --cuda 0\n'.format(ps['dataset'],
                            ps['num_layers'], ps['dropout'], ps['learn_rate'], ps['weight_decay'])
                f.write(command)


def generate_vsgc_pre_search_shells():
    dataset = ['cora']
    num_layers = [2, 4, 8, 16, 24, 32, 40, 48]
    alpha = [1]
    lambd = [1]
    dropout = [0, 0.5, 0.8]
    learn_rate = [0.5, 0.3, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    filename = ['VSGC_Pre_nsl_search']
    id = 0
    with open('../shells/{}_{}.sh'.format(filename[0], '_'.join(dataset)), 'w') as f:
        f.write('#! /bin/bash\n')
        for _ in range(5):
            params = itertools.product(dataset, num_layers, alpha, lambd, dropout, learn_rate, weight_decay, filename)
            for p in params:
                command = 'python train_vsgc_pre_tmp.py --dataset {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 1 --id {}\n'.format(p[0], p[1], p[2], p[3], p[4],
                                                                                     p[5], p[6], p[7], id)
                id += 1
                f.write(command)

    print("{}条命令".format(id))

def generate_vsgc_pre_search_full_shells():
    # dataset = ['cornell', 'texas', 'wisconsin']
    # dataset = ['cora']
    # num_layers = [2, 4, 8, 16, 24, 32, 40, 48]
    dataset = ['wisconsin']
    num_layers = [2, 4, 6, 8]
    alpha = [1]
    lambd = [-0.1, -0.05, -0.01, -0.001, 0, 0.001, 0.01, 0.05, 0.1]
    dropout = [0, 0.5, 0.8]
    learn_rate = [0.5, 0.3, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    filename = ['VSGC_Pre_search_full']
    id = 0
    with open('../shells/{}_{}.sh'.format(filename[0], '_'.join(dataset)), 'w') as f:
        f.write('#! /bin/bash\n')
        for i in range(10):
            params = itertools.product(dataset, num_layers, alpha, lambd, dropout, learn_rate, weight_decay, filename)
            for p in params:
                split = '../data/splits/{}_split_0.6_0.2_{}.npz'.format(p[0], i)
                command = 'python train_vsgc_pre.py --dataset {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 2 --split {} --id {}\n'.format(p[0], p[1], p[2], p[3], p[4],
                                                                 p[5], p[6], p[7], split, id)
                id += 1
                f.write(command)
    print("{}条命令".format(id))


def generate_vsgc_pre_result_shells():
    with open('../shells/VSGC_Pre_result.sh', 'w') as f:
        f.write('#! /bin/bash\n')
        params = []
        # params.append({'dataset': 'cora', 'num_layers': 32, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.8, 'seed': 42, 'learn_rate': 0.3, 'weight_decay': 5e-06, 'num_epochs': 1500, 'patience': 100, 'cuda': 2, 'filename': 'VSGC_Pre_nosl_search', 'split': 'semi', 'train_loss': 0.14258791506290436, 'train_acc': 0.9928571428571429, 'val_loss': 0.7066516280174255, 'val_acc': 0.806, 'test_loss': 0.6727521419525146, 'test_acc': 0.834})
        params.append({'dataset': 'citeseer', 'num_layers': 16, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.5, 'seed': 42, 'learn_rate': 0.1, 'weight_decay': 0.001, 'num_epochs': 1500, 'patience': 100, 'cuda': 0, 'filename': 'VSGC_Pre_search', 'split': 'semi', 'id': 254, 'train_loss': 0.730340301990509, 'train_acc': 0.8833333333333333, 'val_loss': 1.1936839818954468, 'val_acc': 0.732, 'test_loss': 1.1579455137252808, 'test_acc': 0.744})
        params.append({'dataset': 'pubmed', 'num_layers': 40, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.8, 'seed': 42, 'learn_rate': 0.5, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'cuda': 1, 'filename': 'VSGC_Pre_search', 'split': 'semi', 'id': 483, 'train_loss': 0.2379106730222702, 'train_acc': 0.9666666666666667, 'val_loss': 0.5157273411750793, 'val_acc': 0.82, 'test_loss': 0.5398994088172913, 'test_acc': 0.816})

        for ps in params:
            for _ in range(100):
                command = 'python train_vsgc_pre.py --dataset {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename VSGC_Pre_result --cuda 3\n'.format(ps['dataset'],
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
    id = 0
    with open('../shells/{}_{}.sh'.format(filename[0], '_'.join(dataset)), 'w') as f:
        f.write('#! /bin/bash\n')
        for _ in range(5):
            params = itertools.product(dataset, num_k, num_layers, alpha, lambd, dropout, learn_rate, weight_decay, filename)
            for p in params:
                command = 'python train_vsgc.py --dataset {} --num_k {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 0\n'.format(p[0], p[1], p[2], p[3], p[4],
                                                                                     p[5], p[6], p[7], p[8])
                id += 1
                f.write(command)
    print("{}条命令".format(id))


def generate_vsgc_result_shells():
    with open('../shells/VSGC_result.sh', 'w') as f:
        f.write('#! /bin/bash\n')
        params = []
        # params.append({'dataset': 'cora', 'num_hidden': 64, 'num_k': 8, 'num_layers': 1, 'alpha': 1.0, 'lambd': 1.0, 'batch_norm': False, 'residual': False, 'dropout': 0.5, 'seed': 42, 'learn_rate': 0.1, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'cuda': 0, 'filename': 'VSGC_search', 'split': 'semi', 'train_loss': 0.7972564101219177, 'train_acc': 0.9642857142857143, 'val_loss': 1.1745821237564087, 'val_acc': 0.8, 'test_loss': 1.1455549001693726, 'test_acc': 0.836})
        # params.append({'dataset': 'citeseer', 'num_hidden': 64, 'num_k': 2, 'num_layers': 1, 'alpha': 1.0, 'lambd': 1.0, 'batch_norm': False, 'residual': False, 'dropout': 0.0, 'seed': 42, 'learn_rate': 0.1, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'cuda': 1, 'filename': 'VSGC_search', 'split': 'semi', 'train_loss': 1.1990128755569458, 'train_acc': 0.9166666666666666, 'val_loss': 1.5252231359481812, 'val_acc': 0.72, 'test_loss': 1.515850305557251, 'test_acc': 0.725})
        params.append( {'dataset': 'pubmed', 'num_hidden': 64, 'num_k': 32, 'num_layers': 2, 'alpha': 1.0, 'lambd': 1.0, 'batch_norm': False, 'residual': False, 'dropout': 0.5, 'seed': 42, 'learn_rate': 0.01, 'weight_decay': 0.0005, 'num_epochs': 1500, 'patience': 100, 'cuda': 3, 'filename': 'VSGC_search', 'split': 'semi', 'train_loss': 0.07003548741340637, 'train_acc': 0.9833333333333333, 'val_loss': 0.5115864276885986, 'val_acc': 0.826, 'test_loss': 0.5647717118263245, 'test_acc': 0.807})

        for ps in params:
            for _ in range(100):
                command = 'python train_vsgc.py --dataset {} --num_k {} --num_layers {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename VSGC_result --cuda 2\n'.format(ps['dataset'], ps['num_k'],
                            ps['num_layers'], ps['alpha'], ps['lambd'], ps['dropout'], ps['learn_rate'], ps['weight_decay'])
                f.write(command)

def generate_vblockgcn_search_shells():
    dataset = ['cora']
    k = [2, 4, 6, 8, 10, 12]
    num_blocks = [2, 3, 4]
    alpha = [1]
    lambd = [1]
    # dropout = [0, 0.5, 0.8]
    dropout = [0]
    learn_rate = [0.5, 0.3, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    attention = [True]
    filename = ['VBlockGCN_nsl_att_l2_search']
    id = 0
    with open('../shells/{}_{}.sh'.format(filename[0], '_'.join(dataset)), 'w') as f:
        f.write('#! /bin/bash\n')
        for _ in range(5):
            params = itertools.product(dataset, k, num_blocks, alpha, lambd, dropout, learn_rate, weight_decay, filename, attention)
            for p in params:
                if p[9] == False:
                    command = 'python train_block_vgcn_tmp.py --dataset {} --k {} --num_blocks {} --alpha {} --lambd {} --dropout {} ' \
                              '--learn_rate {} --weight_decay {} --filename {} --cuda 2 --id {}\n'.format(p[0], p[1], p[2], p[3], p[4],
                                                                                         p[5], p[6], p[7], p[8], id)
                    id += 1
                else:
                    command = 'python train_block_vgcn_tmp.py --dataset {} --k {} --num_blocks {} --alpha {} --lambd {} --dropout {} ' \
                              '--learn_rate {} --weight_decay {} --filename {} --cuda 0 --attention --id {}\n'.format(p[0], p[1], p[2],
                                                                                                  p[3], p[4],
                                                                                                  p[5], p[6], p[7],
                                                                                                  p[8], id)
                    id += 1
                f.write(command)
        print("{}条命令".format(id))

def generate_vblockgcn_result_shells():
    filename = 'VBlockGCN_nsl_att_result'
    with open('../shells/{}.sh'.format(filename), 'w') as f:
        f.write('#! /bin/bash\n')
        params = []
        params.append({'dataset': 'cora', 'num_hidden': 64, 'k': 4, 'num_blocks': 2, 'alpha': 1.0, 'lambd': 1.0, 'residual': False, 'dropout': 0.8, 'attention': True, 'seed': 42, 'learn_rate': 0.01, 'weight_decay': 0.001, 'num_epochs': 1500, 'patience': 100, 'cuda': 0, 'filename': 'VBlockGCN_nsl_att_search', 'id': 284, 'train_loss': 0.19289906322956085, 'train_acc': 0.9714285714285714, 'val_loss': 0.6229188442230225, 'val_acc': 0.798, 'test_loss': 0.577835202217102, 'test_acc': 0.827})
        params.append({'dataset': 'citeseer', 'num_hidden': 64, 'k': 10, 'num_blocks': 2, 'alpha': 1.0, 'lambd': 1.0, 'residual': False, 'dropout': 0.5, 'attention': True, 'seed': 42, 'learn_rate': 0.01, 'weight_decay': 0.01, 'num_epochs': 1500, 'patience': 100, 'cuda': 1, 'filename': 'VBlockGCN_nsl_att_search', 'id': 2203, 'train_loss': 0.9902288913726807, 'train_acc': 0.8666666666666667, 'val_loss': 1.2891196012496948, 'val_acc': 0.72, 'test_loss': 1.2846444845199585, 'test_acc': 0.739})
        params.append({'dataset': 'pubmed', 'num_hidden': 64, 'k': 6, 'num_blocks': 2, 'alpha': 1.0, 'lambd': 1.0, 'residual': False, 'dropout': 0.5, 'attention': True, 'seed': 42, 'learn_rate': 0.3, 'weight_decay': 0.001, 'num_epochs': 1500, 'patience': 100, 'cuda': 2, 'filename': 'VBlockGCN_nsl_att_search', 'id': 4352, 'train_loss': 0.15525469183921814, 'train_acc': 0.9833333333333333, 'val_loss': 0.49483782052993774, 'val_acc': 0.806, 'test_loss': 0.5370917320251465, 'test_acc': 0.811})

        for ps in params:
            for _ in range(100):
                command = 'python train_block_vgcn_tmp.py --dataset {} --num_hidden {} --k {} --num_blocks {} --alpha {} --lambd {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 0 --attention\n'.format(ps['dataset'], ps['num_hidden'], ps['k'], ps['num_blocks'],
                            ps['alpha'], ps['lambd'], ps['dropout'], ps['learn_rate'],ps['weight_decay'], filename)
                f.write(command)

def generate_mlp_search_full_shells():
    # dataset = ['cornell', 'texas', 'wisconsin']
    # dataset = ['cora']
    # num_layers = [2, 4, 8, 16, 24, 32, 40, 48]
    dataset = ['cornell', 'texas', 'wisconsin']
    num_layers = [2]
    dropout = [0, 0.5, 0.8]
    learn_rate = [0.5, 0.3, 0.1, 0.01]
    weight_decay = [0, 1e-2, 1e-3, 5e-4, 5e-5, 5e-6]
    filename = ['MLP_search_full']
    id = 0
    with open('../shells/{}_{}.sh'.format(filename[0], '_'.join(dataset)), 'w') as f:
        f.write('#! /bin/bash\n')
        for i in range(10):
            params = itertools.product(dataset, num_layers, dropout, learn_rate, weight_decay, filename)
            for p in params:
                split = '../data/splits/{}_split_0.6_0.2_{}.npz'.format(p[0], i)
                command = 'python train_mlp.py --dataset {} --num_layers {} --dropout {} ' \
                          '--learn_rate {} --weight_decay {} --filename {} --cuda 2 --split {} --id {}\n'.format(p[0], p[1], p[2], p[3], p[4],
                                                                 p[5], split, id)
                id += 1
                f.write(command)
    print("{}条命令".format(id))

if __name__ == '__main__':
    generate_vblockgcn_search_shells()