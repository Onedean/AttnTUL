import warnings
import time
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import EarlyStopping, load_data, accuracy_1, accuracy_5, loss_with_earlystop_plot
from datasets import get_dataset, get_dataloader
from models import MolNet, GcnNet
from tqdm import tqdm


def parse_args():
    """ This is a function used to parse command line arguments

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='Moving Object Linking')
    parse.add_argument('--dataset', type=str,
                       default="shenzhen-mini", help='Dataset for training')
    parse.add_argument('--read_pkl', type=bool, default=True,
                       help='Read preprocessed input')
    parse.add_argument('--times', type=int, default=10,
                       help='times of repeat experiment')
    parse.add_argument('--epochs', type=int, default=80,
                       help='Number of epochs to train')
    parse.add_argument('--train_batch', type=int,
                       default=16, help='Size of train batch')
    parse.add_argument('--valid_batch', type=int,
                       default=16, help='Size of valid batch')
    parse.add_argument('--test_batch', type=int, default=16,
                       help='Size of test batch')
    parse.add_argument('--patience', type=int, default=10,
                       help='Number of early stop patience')

    parse.add_argument('--gcn_lr', type=float, default=1e-2,
                       help='Initial gcn learning rate')
    parse.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay (L2 loss on parameters)')
    parse.add_argument('--localGcn_hidden', type=int,
                       default=512, help='Number of local gcn hidden units')
    parse.add_argument('--globalGcn_hidden', type=int,
                       default=512, help='Number of global gcn hidden units')
    parse.add_argument('--gcn_dropout', type=float, default=0.1,
                       help='Dropout rate (1 - keep probability)')

    parse.add_argument('--Attn_Strategy', type=str,
                       default='cos', help='Global Attention Strategy')
    parse.add_argument('--Softmax_Strategy', type=str,
                       default='complex', help='Global Softmax Strategy')
    parse.add_argument('--Pool_Strategy', type=str,
                       default='max', help='Pooling layer Strategy')

    parse.add_argument('--merge_use', type=bool, default=True,
                       help='Merge same adjacent edges or not')
    parse.add_argument('--state_use', type=bool, default=False,
                       help='Fusion state information or not')
    parse.add_argument('--time_use', type=bool, default=True,
                       help='Fusion time information or not')

    parse.add_argument('--grid_size', type=int,
                       default=120, help='Size of grid')

    parse.add_argument('--d_model', type=int, default=128,
                       help='Number of point vector dim')
    parse.add_argument('--encode_lr', type=float, default=1e-3,
                       help='Initial encode learning rate')
    parse.add_argument('--d_k', type=int, default=64,
                       help='Number of querry vector dim')
    parse.add_argument('--d_v', type=int, default=64,
                       help='Number of key vector dim')
    parse.add_argument('--d_ff', type=int, default=512,
                       help='Number of Feed forward transform dim')
    parse.add_argument('--n_heads', type=int, default=5,
                       help='Number of heads')
    parse.add_argument('--n_layers', type=int, default=3,
                       help='Number of EncoderLayer')

    args = parse.parse_args()
    return args


def train_model(epochs, patience, train_dataset, train_batch, test_dataset, valid_batch, valid_sampler, LocalGcnModel, GlobalGcnModel, MolModel, optimizer_localgcn, optimizer_globalgcn, optimizer_mol, local_feature, local_adj, global_feature, global_adj, device):
    """[This is a function used to train and verify the model. It uses the early stop method to record the best checkpoint]

    Args:
        epochs ([int]): [Number of repeated rounds of single experiment]
        patience ([int]): [Patience of early stop method]
        train_dataset ([obejct]): [training set]
        train_batch ([int]): [batch size of training set]
        test_dataset ([obejct]): [test datset]
        valid_batch ([int]): [batch size of validation set]
        valid_sampler ([obejct]): [validation set sampler]
        LocalGcnModel ([obejct]): [local gcn net]
        GlobalGcnModel ([obejct]): [global gcn net]
        MolModel ([obejct]): [stan net]
        local_feature ([torch.tensor]): [local graph feature]
        local_adj ([torch.tensor]): [local graph adj]
        global_feature ([torch.tensor]): [global graph feature]
        global_adj ([torch.tensor]): [global graph adj]
        device ([torch.tensor]): [Use GPU server]

    Returns:
        avg_train_losses ([list]), avg_valid_losses ([list]): [Return the records of loss in the training process and verification process, and draw the loss decline curve]
    """
    avg_train_losses = []
    avg_valid_losses = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):

        loss_train, y_predict_list, y_true_list, acc1_list, acc5_list = 0, [], [], [], []
        LocalGcnModel.train()
        GlobalGcnModel.train()
        MolModel.train()
        grid_emb = LocalGcnModel(local_feature, local_adj)
        traj_emb = GlobalGcnModel(global_feature, global_adj)
        # for input_seq, input_index, y_true in get_dataloader(True, train_dataset, train_batch):
        for input_seq, time_seq, state_seq, input_index, y_true in get_dataloader(True, train_dataset, train_batch):
            # input_seq, y_true = input_seq.to(device), y_true.to(device)
            input_seq, time_seq, state_seq, y_true = input_seq.to(
                device), time_seq.to(device), state_seq.to(device),  y_true.to(device)
            # y_predict = MolModel(grid_emb, traj_emb, input_seq, input_index)
            y_predict = MolModel(grid_emb, traj_emb, input_seq,
                                 time_seq, state_seq, input_index)

            y_predict_list.extend(torch.max(y_predict, 1)[
                                  1].cpu().numpy().tolist())
            y_true_list.extend(y_true.cpu().numpy().tolist())
            acc1_list.append(accuracy_1(y_predict, y_true))
            acc5_list.append(accuracy_5(y_predict, y_true))
            loss_train += F.nll_loss(y_predict, y_true)

        loss_train_sum = loss_train.item()
        p = precision_score(y_true_list, y_predict_list, average='macro')
        r = recall_score(y_true_list, y_predict_list, average='macro')
        f1 = f1_score(y_true_list, y_predict_list, average='macro')

        optimizer_localgcn.zero_grad()
        optimizer_globalgcn.zero_grad()
        optimizer_mol.zero_grad()
        loss_train.backward()
        optimizer_localgcn.step()
        optimizer_globalgcn.step()
        optimizer_mol.step()

        print('Epoch: {}/{}'.format(epoch, epochs))
        print('train_loss:{:.5f}  acc1:{:.4f}  acc5:{:.4f}  Macro-P:{:.4f}  Macro-R:{:.4f}  Macro-F1:{:.4f}'.format(
            loss_train_sum, np.mean(acc1_list), np.mean(acc5_list), p, r, f1))

        loss_valid, y_predict_list, y_true_list, acc1_list, acc5_list = 0, [], [], [], []
        LocalGcnModel.eval()
        GlobalGcnModel.eval()
        MolModel.eval()
        with torch.no_grad():
            grid_emb = LocalGcnModel(local_feature, local_adj)
            traj_emb = GlobalGcnModel(global_feature, global_adj)
            # for input_seq, input_index, y_true in get_dataloader(False, test_dataset, valid_batch, valid_sampler):
            for input_seq, time_seq, state_seq, input_index, y_true in get_dataloader(False, test_dataset, valid_batch, valid_sampler):
                # input_seq, y_true = input_seq.to(device), y_true.to(device)
                input_seq, time_seq, state_seq, y_true = input_seq.to(
                    device), time_seq.to(device), state_seq.to(device),  y_true.to(device)
                # y_predict = MolModel(grid_emb, traj_emb, input_seq, input_index)
                y_predict = MolModel(
                    grid_emb, traj_emb, input_seq, time_seq, state_seq, input_index)

                y_predict_list.extend(torch.max(y_predict, 1)[
                                      1].cpu().numpy().tolist())
                y_true_list.extend(y_true.cpu().numpy().tolist())
                acc1_list.append(accuracy_1(y_predict, y_true))
                acc5_list.append(accuracy_5(y_predict, y_true))
                loss_valid += F.nll_loss(y_predict, y_true)

        loss_valid_sum = loss_valid.item()
        p = precision_score(y_true_list, y_predict_list, average='macro')
        r = recall_score(y_true_list, y_predict_list, average='macro')
        f1 = f1_score(y_true_list, y_predict_list, average='macro')
        print('valid_loss:{:.5f}  acc1:{:.4f}  acc5:{:.4f}  Macro-P:{:.4f}  Macro-R:{:.4f}  Macro-F1:{:.4f}'.format(
            loss_valid_sum, np.mean(acc1_list), np.mean(acc5_list), p, r, f1))

        avg_train_losses.append(loss_train_sum)
        avg_valid_losses.append(loss_valid_sum)

        # early_stopping(loss_valid_sum, (LocalGcnModel, GlobalGcnModel, MolModel))
        early_stopping(-np.mean(acc1_list),
                       (LocalGcnModel, GlobalGcnModel, MolModel))

        if early_stopping.early_stop:
            print('Early Stop!')
            break

    return avg_train_losses, avg_valid_losses


def test_model(test_dataset, test_batch, test_sampler, LocalGcnModel, GlobalGcnModel, MolModel, local_feature, local_adj, global_feature, global_adj, device):
    """[summary]
    
    Args:
        test_dataset ([obejct]): [test datset]
        test_batch ([int]): [batch size of test set]
        test_sampler ([object]): [test set sampler]
        LocalGcnModel ([obejct]): [local gcn net]
        GlobalGcnModel ([obejct]): [global gcn net]
        MolModel ([obejct]): [stan net]
        local_feature ([torch.tensor]): [local graph feature]
        local_adj ([torch.tensor]): [local graph adj]
        global_feature ([torch.tensor]): [global graph feature]
        global_adj ([torch.tensor]): [global graph adj]
        device ([torch.tensor]): [Use GPU server]
    """
    loss_test, y_predict_list, y_true_list, acc1_list, acc5_list = 0, [], [], [], []
    LocalGcnModel.eval()
    GlobalGcnModel.eval()
    MolModel.eval()
    with torch.no_grad():
        grid_emb = LocalGcnModel(local_feature, local_adj)
        traj_emb = GlobalGcnModel(global_feature, global_adj)

        # for input_seq, input_index, y_true in get_dataloader(False, test_dataset, test_batch, test_sampler):
        for input_seq, time_seq, state_seq, input_index, y_true in get_dataloader(False, test_dataset, test_batch, test_sampler):
            # input_seq, y_true = input_seq.to(device), y_true.to(device)
            input_seq, time_seq, state_seq, y_true = input_seq.to(
                device), time_seq.to(device), state_seq.to(device), y_true.to(device)
            # y_predict = MolModel(grid_emb, traj_emb, input_seq, input_index)
            y_predict = MolModel(grid_emb, traj_emb, input_seq,
                                 time_seq, state_seq, input_index)

            y_predict_list.extend(torch.max(y_predict, 1)[
                                  1].cpu().numpy().tolist())
            y_true_list.extend(y_true.cpu().numpy().tolist())
            acc1_list.append(accuracy_1(y_predict, y_true))
            acc5_list.append(accuracy_5(y_predict, y_true))
            loss_test += F.nll_loss(y_predict, y_true)

    loss_test_sum = loss_test.item()
    p = precision_score(y_true_list, y_predict_list, average='macro')
    r = recall_score(y_true_list, y_predict_list, average='macro')
    f1 = f1_score(y_true_list, y_predict_list, average='macro')
    print('test_loss:{:.5f}  acc1:{:.4f}  acc5:{:.4f}  Macro-P:{:.4f}  Macro-R:{:.4f}  Macro-F1:{:.4f}'.format(
        loss_test_sum, np.mean(acc1_list), np.mean(acc5_list), p, r, f1))


def main(dataset, read_pkl, times, epochs, train_batch, valid_batch, test_batch, patience, gcn_lr, weight_decay, localGcn_hidden, globalGcn_hidden, gcn_dropout, encode_lr, d_model, d_k, d_v, d_ff, n_heads, n_layers, Attn_Strategy, Softmax_Strategy, Pool_Strategy, merge_use, state_use, time_use, grid_size):
    """[This is the entry function for the experiment]

    Args:
        dataset ([str]): [dataset name]
        read_pkl ([bool]): [Whether to use preprocessed files directly]
        times ([int]): [Number of experimental repetitions]
        epochs ([type]): [Number of repeated rounds of single experiment]
        train_batch ([type]): [Batch size of training set]
        valid_batch ([type]): [Batch size of valid set]
        test_batch ([type]): [Batch size of test set]
        patience ([type]): [Patience of early stop method]
        gcn_lr ([type]): [gcn lr]
        weight_decay ([type]): [l2 value]
        localGcn_hidden ([type]): [local gcn hidden dim]]
        globalGcn_hidden ([type]): [global gcn hidden dim]
        gcn_dropout ([type]): [gcn dropout value]
        encode_lr ([type]): [encoder lr]
        d_model ([type]): [traj embedding dim]
        d_k ([type]): [key dim]
        d_v ([type]): [value dim]
        d_ff ([type]): [ffn dim]
        n_heads ([type]): [the number of heads]
        n_layers ([type]): [the number of self-attention layers]
        Attn_Strategy ([type]): [What kind of attention strategy are used]
        Softmax_Strategy ([type]): [What kind of softmax strategy are used]
        Pool_Strategy ([type]): [What kind of pooling strategy are used]
        merge_use ([type]): [Use merge window or not]
        state_use ([type]): [Use state information or not]
        time_use ([type]): [Use time information or not]
        grid_size ([type]): [the size of single grid]
    """
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    local_feature, local_adj, global_feature, global_adj, user_traj_train, user_traj_test, grid_nums, traj_nums, user_nums, test_nums = load_data(
        dataset, read_pkl, grid_size)
    local_feature, local_adj, global_feature, global_adj = local_feature.to(
        device), local_adj.to(device), global_feature.to(device), global_adj.to(device)

    train_dataset, test_dataset, valid_sampler, test_sampler = get_dataset(
        test_nums, user_traj_train, user_traj_test)

    for idx, seed in enumerate(random.sample(range(0, 1000), times)):

        # Fixed random seed(random seed 555 can be used to reproduce the results of this paper)
        seed = 555
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Initialization model
        LocalGcnModel = GcnNet(grid_nums, localGcn_hidden,
                               d_model, gcn_dropout).to(device)
        GlobalGcnModel = GcnNet(
            grid_nums, globalGcn_hidden, d_model, gcn_dropout).to(device)
        MolModel = MolNet(Attn_Strategy, Softmax_Strategy, Pool_Strategy,
                          d_model, d_k, d_v, d_ff, n_heads, n_layers, user_nums).to(device)

        # Initialize optimizer
        optimizer_localgcn = torch.optim.Adam(
            LocalGcnModel.parameters(), lr=gcn_lr, weight_decay=weight_decay)
        optimizer_globalgcn = torch.optim.Adam(
            GlobalGcnModel.parameters(), lr=gcn_lr, weight_decay=weight_decay)
        optimizer_mol = torch.optim.Adam(MolModel.parameters(), lr=encode_lr)

        # Train model
        print('The {} round, start training with random seed {}'.format(idx, seed))
        t_total = time.time()

        avg_train_losses, avg_valid_losses = train_model(epochs, patience, train_dataset, train_batch, test_dataset, valid_batch, valid_sampler, LocalGcnModel,
                                                         GlobalGcnModel, MolModel, optimizer_localgcn, optimizer_globalgcn, optimizer_mol, local_feature, local_adj, global_feature, global_adj, device)

        loss_with_earlystop_plot(avg_train_losses, avg_valid_losses)

        LocalGcnModel.load_state_dict(torch.load('../temp/checkpoint0.pt'))
        GlobalGcnModel.load_state_dict(torch.load('../temp/checkpoint1.pt'))
        MolModel.load_state_dict(torch.load('../temp/checkpoint2.pt'))

        test_model(test_dataset, test_batch, test_sampler, LocalGcnModel, GlobalGcnModel,
                   MolModel, local_feature, local_adj, global_feature, global_adj, device)

        print(f"Total time elapsed: {time.time() - t_total:.4f}s")
        print('Fininsh trainning in seed {}\n'.format(seed))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_args()
    main(dataset=args.dataset, read_pkl=args.read_pkl, times=args.times, epochs=args.epochs, train_batch=args.train_batch, valid_batch=args.valid_batch,
         test_batch=args.test_batch, patience=args.patience, gcn_lr=args.gcn_lr, weight_decay=args.weight_decay, localGcn_hidden=args.localGcn_hidden,
         globalGcn_hidden=args.globalGcn_hidden, gcn_dropout=args.gcn_dropout, encode_lr=args.encode_lr, d_model=args.d_model, d_k=args.d_k, d_v=args.d_v,
         d_ff=args.d_ff, n_heads=args.n_heads, n_layers=args.n_layers, Attn_Strategy=args.Attn_Strategy, Softmax_Strategy=args.Softmax_Strategy,
         Pool_Strategy=args.Pool_Strategy, merge_use=args.merge_use, state_use=args.state_use, time_use=args.time_use, grid_size=args.grid_size)
