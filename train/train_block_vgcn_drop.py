import sys

from utils.data_geom import load_data_from_file

sys.path.append('../')
import argparse
import time
import torch.nn.functional as F
from nets.vgcn_block_net_drop import VGCNBlockNet
from train.early_stopping import EarlyStopping
from train.metrics import evaluate_acc_loss
from train.train_gcn import set_seed
from utils.data_mine import load_data_default, load_data_mine
import torch as th
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cornell')
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--lambd', type=float, default=1)
    parser.add_argument('--feat_drop', type=float, default=0)
    parser.add_argument('--attention', action='store_true', default=True)
    parser.add_argument('--edge_drop', type=float, default=0)
    parser.add_argument('--important', action='store_true', default=False)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learn_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=1500)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--filename', type=str, default='VBlockGCN')
    # parser.add_argument('--split', type=str, default='semi')
    parser.add_argument('--split', type=str, default='../data/splits/cornell_split_0.6_0.2_0.npz')
    parser.add_argument('--id', type=int, default=0)
    args = parser.parse_args()

    print("attention:{}".format(args.attention))

    if args.split != 'semi':
        graph, features, labels, train_mask, val_mask, test_mask, num_feats, num_classes = load_data_from_file(
            args.dataset, splits_file_path=args.split)
    else:
        graph, features, labels, train_mask, val_mask, test_mask, num_feats, num_classes = load_data_default(
            args.dataset)
    labels = labels.squeeze()
    model = VGCNBlockNet(num_feats, num_classes, args.k, args.num_blocks, alpha=args.alpha, lambd=args.lambd,
                         feat_drop=args.feat_drop, attention=args.attention, edge_drop=args.edge_drop,
                         important=args.important)

    # set_seed(args.seed)

    optimizer = th.optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=args.weight_decay)
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
    print(model)
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
        print("Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".
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