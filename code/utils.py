import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from rawprocess import get_data_and_graph


class EarlyStopping:
    """[Early stops the training if validation loss doesn't improve after a given patience.]
    """

    def __init__(self, logger, patience=7, verbose=False, delta=0):
        """[Receive optional parameters]

        Args:
            patience (int, optional): [How long to wait after last time validation loss improved.]. Defaults to 7.
            verbose (bool, optional): [If True, prints a message for each validation loss improvement. ]. Defaults to False.
            delta (int, optional): [Minimum change in the monitored quantity to qualify as an improvement.]. Defaults to 0.
        """
        self.logger = logger
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, all_model):
        """[this is a Callback function]

        Args:
            val_loss ([float]): [The loss of receiving verification was changed to accuracy as the stop criterion in our experiment]
            all_model ([list]): [Accept the list of all models to save]
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, all_model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, all_model)
            self.counter = 0

    def save_checkpoint(self, val_loss, all_model):
        """[Saves model when validation loss decrease.]

        Args:
            val_loss ([type]): [The loss value corresponding to the best checkpoint needs to be saved]
            all_model ([list]): [Save the list of models corresponding to the best checkpoint]
        """
        if self.verbose:
            self.logger.info(
                f'Validation acc increased ({-self.val_loss_min:.6f} --> {-val_loss:.6f}).  Saving model ...')
        for idx, model in enumerate(all_model):
            # The parameters of the optimal model so far will be stored here
            torch.save(model.state_dict(), '../temp/checkpoint'+str(idx)+'.pt')
        self.val_loss_min = val_loss


def load_data(dataset, read_pkl, grid_size):
    """[This is a function used to load data]

    Args:
        dataset ([str]): [The name of the dataset]
        read_pkl ([bool]): [Whether to use the preprocessed data file]
        grid_size ([int]): [Grid size of preprocessing trajectory data division]

    Returns:
        [type]: [description]
    """
    if read_pkl == False:
        if dataset == 'gowalla-mini':
            raw_path = '../data/gowalla/raw/gowalla-mini.csv'
        elif dataset == 'gowalla-all':
            raw_path = '../data/gowalla/raw/gowalla-all.csv'
        elif dataset == 'shenzhen-mini':
            raw_path = '../data/shenzhen/raw/shenzhen-mini.csv'
        elif dataset == 'shenzhen-min':
            raw_path = '../data/shenzhen/raw/shenzhen-min.csv'
        elif dataset == 'shenzhen-all':
            raw_path = '../data/shenzhen/raw/shenzhen-all.csv'
        elif dataset == 'geolife-mini':
            raw_path = '../data/geolife/raw/geolife-mini.csv'
        elif dataset == 'geolife-all':
            raw_path = '../data/geolife/raw/geolife-all.csv'
        else:
            print('The dataset does not exist')
            exit()
        return get_data_and_graph(raw_path, read_pkl, grid_size)
    else:
        if dataset == 'gowalla-mini':
            raw_path = '../data/gowalla/process/gowalla-mini-' + \
                str(grid_size)+'.pkl'
        elif dataset == 'gowalla-all':
            raw_path = '../data/gowalla/process/gowalla-all-' + \
                str(grid_size)+'.pkl'
        elif dataset == 'shenzhen-mini':
            raw_path = '../data/shenzhen/process/shenzhen-mini-' + \
                str(grid_size)+'.pkl'
        elif dataset == 'shenzhen-min':
            raw_path = '../data/shenzhen/process/shenzhen-min-' + \
                str(grid_size)+'.pkl'
        elif dataset == 'shenzhen-all':
            raw_path = '../data/shenzhen/process/shenzhen-all-' + \
                str(grid_size)+'.pkl'
        elif dataset == 'geolife-mini':
            raw_path = '../data/geolife/process/geolife-mini-' + \
                str(grid_size)+'.pkl'
        elif dataset == 'geolife-all':
            raw_path = '../data/geolife/process/geolife-all-' + \
                str(grid_size)+'.pkl'
        else:
            print('The dataset does not exist')
            exit()
        f = open(raw_path, 'rb')
        local_feature, local_adj, global_feature, global_adj, user_traj_train, user_traj_test, grid_nums, traj_nums, user_nums, test_nums = pickle.load(
            f)
        f.close()
        return local_feature, local_adj, global_feature, global_adj, user_traj_train, user_traj_test, grid_nums, traj_nums, user_nums, test_nums


def accuracy_1(pred, targ):
    """[Used to calculate trajectory links acc@1]

    Args:
        pred ([torch.tensor]): [Predicted user probability distribution]
        targ ([type]): [The real label of the user corresponding to the trajectory]

    Returns:
        [float]: [acc@1]
    """
    pred = torch.max(torch.log_softmax(pred, dim=1), 1)[1]
    ac = ((pred == targ).float()).sum().item() / targ.size()[0]
    return ac


def accuracy_5(pred, targ):
    """[Used to calculate trajectory links acc@5]

    Args:
        pred ([torch.tensor]): [Predicted user probability distribution]
        targ ([type]): [The real label of the user corresponding to the trajectory]

    Returns:
        [float]: [acc@5]
    """
    pred = torch.topk(torch.log_softmax(pred, dim=1), k=5,
                      dim=1, largest=True, sorted=True)[1]
    ac = (torch.tensor([t in p for p, t in zip(pred, targ)]
                       ).float()).sum().item() / targ.size()[0]
    return ac


def loss_plot(train_loss, val_loss):
    """[Function used to plot the loss curve]

    Args:
        train_loss ([list]): [Loss list of training sets]
        val_loss ([list]): [Loss list of Validation sets]
    """
    plt.switch_backend('agg')
    plt.plot(range(len(train_loss)), train_loss, label='Train loss', linewidth=2,
             color='orange', marker='o', markerfacecolor='r', markersize=5)
    plt.plot(range(len(val_loss)), val_loss, label='Validation loss',
             linewidth=2, color='blue', marker='o', markerfacecolor='r', markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Loss-fig')
    plt.legend()
    plt.savefig('../log/Loss.png')


def loss_with_earlystop_plot(avg_train_losses, avg_valid_losses):
    """[Function used to plot the loss curve and early stop line]

    Args:
        train_loss ([list]): [Loss list of training sets]
        val_loss ([list]): [Loss list of Validation sets]
    """
    # visualize the loss as the network trained
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(avg_train_losses)+1),
             avg_train_losses, label='Training Loss')
    plt.plot(range(1, len(avg_valid_losses)+1),
             avg_valid_losses, label='Validation Loss')

    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses))+1
    plt.axvline(minposs, linestyle='--', color='r',
                label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')

    # plt.ylim(0, 10) # consistent scale
    plt.xlim(0, len(avg_train_losses)+1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('../log/early_stop_loss.png')


def acc_plot(train_acc1, val_acc1, train_acc5, val_acc5):
    """[Accuracy curve drawing function]

    Args:
        train_acc1 ([list]): [Training set acc@1 list]
        val_acc1 ([list]): [Validation set acc@1 list]
        train_acc5 ([list]): [Training set acc@5 list]
        val_acc5 ([list]): [Validation set acc@5 list]
    """
    plt.switch_backend('agg')
    plt.figure(figsize=(30, 10), dpi=80)
    plt.figure(1)
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(range(len(train_acc1)), train_acc1, label='Train Acc@1',
             linewidth=2, color='orange', marker='o', markerfacecolor='r', markersize=5)
    plt.plot(range(len(val_acc1)), val_acc1, label='Validation Acc@1',
             linewidth=2, color='blue', marker='o', markerfacecolor='r', markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Acc1-fig')
    plt.legend()
    ax2 = plt.subplot(1, 2, 2)
    plt.plot(range(len(train_acc5)), train_acc5, label='Train Acc@5',
             linewidth=2, color='orange', marker='o', markerfacecolor='r', markersize=5)
    plt.plot(range(len(val_acc5)), val_acc5, label='Validation Acc@5',
             linewidth=2, color='blue', marker='o', markerfacecolor='r', markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Acc5-fig')
    plt.legend()
    plt.savefig('../log/Acc.png')


def macro_plot(train_macro_p, val_macro_p, train_macro_r, val_macro_r, train_macro_f1, val_macro_f1):
    """[macro-p, macro-r and macro-f1 curve drawing of training set and verification set]

    Args:
        train_macro_p ([list]): [Training set macro_p list]
        val_macro_p ([list]): [Validation set macro_p list]
        train_macro_r ([list]): [Training set macro_r list]
        val_macro_r ([list]): [Validation set macro_r list]
        train_macro_f1 ([list]): [Training set macro_f1 list]
        val_macro_f1 ([list]): [Validation set macro_f1 list]
    """
    plt.switch_backend('agg')
    plt.figure(figsize=(45, 10), dpi=80)
    plt.figure(1)
    ax1 = plt.subplot(131)
    plt.plot(range(len(train_macro_p)), train_macro_p, label='Train Acc@1',
             linewidth=2, color='orange', marker='o', markerfacecolor='r', markersize=5)
    plt.plot(range(len(val_macro_p)), val_macro_p, label='Validation Acc@1',
             linewidth=2, color='blue', marker='o', markerfacecolor='r', markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Macro-P value')
    plt.title('Macro-P fig')
    plt.legend()
    ax2 = plt.subplot(132)
    plt.plot(range(len(train_macro_r)), train_macro_r, label='Train Acc@5',
             linewidth=2, color='orange', marker='o', markerfacecolor='r', markersize=5)
    plt.plot(range(len(val_macro_r)), val_macro_r, label='Validation Acc@5',
             linewidth=2, color='blue', marker='o', markerfacecolor='r', markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Macro-R value')
    plt.title('Macro-R fig')
    plt.legend()
    ax3 = plt.subplot(133)
    plt.plot(range(len(train_macro_f1)), train_macro_f1, label='Train macro-F1',
             linewidth=2, color='orange', marker='o', markerfacecolor='r', markersize=5)
    plt.plot(range(len(val_macro_f1)), val_macro_f1, label='Validation macro-F1',
             linewidth=2, color='blue', marker='o', markerfacecolor='r', markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Macro-F1 value')
    plt.title('Macro_F1 fig')
    plt.legend()
    plt.savefig('../log/Macro.png')
