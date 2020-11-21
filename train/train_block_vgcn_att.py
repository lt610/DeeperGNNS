import sys
sys.path.append('../')
from utils.data_geom import load_data_from_file
import argparse
import time
import torch.nn.functional as F
from nets.vgcn_block_net_att import VGCNBlockNet
from train.early_stopping import EarlyStopping
from train.metrics import evaluate_acc_loss
from train.train_gcn import set_seed
from utils.data_mine import load_data_default, load_data_mine
import torch as th
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--lambd', type=float, default=1)
    parser.add_argument('--feat_drop', type=float, default=0.8)
    parser.add_argument('--attention', action='store_true', default=True)
    parser.add_argument('--att_drop', type=float, default=0.5)
    parser.add_argument('--share', action='store_true', default=False)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learn_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=1500)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--filename', type=str, default='VBlockGCN')
    parser.add_argument('--split', type=str, default='semi')
    # parser.add_argument('--split', type=str, default='../data/splits/cora_split_0.6_0.2_0.npz')
    parser.add_argument('--id', type=int, default=0)
    args = parser.parse_args()

    test_print = False
    print("attention:{}".format(args.attention))

    if args.split != 'semi':
        graph, features, labels, train_mask, val_mask, test_mask, num_feats, num_classes = load_data_from_file(
            args.dataset, splits_file_path=args.split)
    else:
        graph, features, labels, train_mask, val_mask, test_mask, num_feats, num_classes = load_data_default(
            args.dataset)
    best_params = None
    if not args.share:
        parms = {'cora': {'dataset': 'cora', 'k': 8, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.8, 'seed': 42, 'learn_rate': 0.5, 'weight_decay': 5e-06, 'num_epochs': 1500, 'patience': 100, 'cuda': 2, 'filename': 'VSGC_Pre_nosl_search', 'split': 'semi', 'train_loss': 0.1324082762002945, 'train_acc': 0.9928571428571429, 'val_loss': 0.6973584890365601, 'val_acc': 0.808, 'test_loss': 0.6653844118118286, 'test_acc': 0.842},
                 'citeseer': {'dataset': 'citeseer', 'k': 16, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.5, 'seed': 42, 'learn_rate': 0.3, 'weight_decay': 5e-05, 'num_epochs': 1500, 'patience': 100, 'cuda': 2, 'filename': 'VSGC_Pre_nsl_search', 'split': 'semi', 'id': 250, 'train_loss': 0.7307705283164978, 'train_acc': 0.95, 'val_loss': 1.2479931116104126, 'val_acc': 0.738, 'test_loss': 1.2341203689575195, 'test_acc': 0.728},
                 'pubmed': {'dataset': 'pubmed', 'k': 8, 'alpha': 1.0, 'lambd': 1.0, 'dropout': 0.8, 'seed': 42, 'learn_rate': 0.1, 'weight_decay': 5e-05, 'num_epochs': 1500, 'patience': 100, 'cuda': 3, 'filename': 'VSGC_Pre_nsl_search', 'split': 'semi', 'id': 208, 'train_loss': 0.289806991815567, 'train_acc': 0.9833333333333333, 'val_loss': 0.62383633852005, 'val_acc': 0.812, 'test_loss': 0.6315034031867981, 'test_acc': 0.792},
                 }
        best_params = parms[args.dataset]

    model = VGCNBlockNet(num_feats, num_classes, args.k, args.num_blocks, alpha=args.alpha, lambd=args.lambd,
                         feat_drop=args.feat_drop, attention=args.attention, att_drop=args.att_drop,
                         best_params=best_params)

    labels = labels.squeeze()
    # set_seed(args.seed)
    if args.share:
        optimizer = th.optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=args.weight_decay)
    else:
        optimizer = th.optim.Adam([{'params': model.mlp1.parameters(), 'lr': best_params['learn_rate'], 'weight_decay': best_params['weight_decay']},
                               {'params': model.mlp2.parameters(), 'lr': args.learn_rate, 'weight_decay': args.weight_decay}])

    early_stopping = EarlyStopping(args.patience, file_name='{}_{}'.format(args.filename, args.dataset))
    # early_stopping = EarlyStoppingBoth()

    device = th.device("cuda:{}".format(args.cuda) if th.cuda.is_available() else "cpu")

    graph = graph.remove_self_loop()

    graph = graph.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    model = model.to(device)

    dur = []
    for epoch in range(args.num_epochs):
        if epoch >= 3:
            t0 = time.time()
        model.train()
        logits = model(graph, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch >= 3:
            dur.append(time.time() - t0)

        train_loss, train_acc = evaluate_acc_loss(model, graph, features, labels, train_mask)
        val_loss, val_acc = evaluate_acc_loss(model, graph, features, labels, val_mask)
        if test_print:
            test_loss, test_acc = evaluate_acc_loss(model, graph, features, labels, test_mask)
            print("Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Test_Loss "
                  "{:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".
                  format(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, np.mean(dur)))
        else:
            print(
                "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {"
                ":.4f}".
                format(epoch, train_loss, train_acc, val_loss, val_acc, np.mean(dur)))

        early_stopping(-val_loss, model)
        if early_stopping.is_stop:
            print("Early stopping")
            model.load_state_dict(early_stopping.load_checkpoint())
            break
    train_loss, train_acc = evaluate_acc_loss(model, graph, features, labels, train_mask)
    val_loss, val_acc = evaluate_acc_loss(model, graph, features, labels, val_mask)
    test_loss, test_acc = evaluate_acc_loss(model, graph, features, labels, test_mask)
    print("Train Loss {:.4f} | Train Acc {:.4f}".format(train_loss, train_acc))
    print("Val Loss {:.4f} | Val Acc {:.4f}".format(val_loss, val_acc))
    print("Test Loss {:.4f} | Test Acc {:.4f}".format(test_loss, test_acc))

    params_results = vars(args)
    params_results['train_loss'] = train_loss
    params_results['train_acc'] = train_acc
    params_results['val_loss'] = val_loss
    params_results['val_acc'] = val_acc
    params_results['test_loss'] = test_loss
    params_results['test_acc'] = test_acc
    filename = '../result/train_result/{}_{}.txt'.format(args.filename, args.dataset)
    with open(filename, 'a') as f:
        f.write(str(params_results) + ', ')
