import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class MolDataset(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, data):
        self.input_seq, self.time_seq, self.state_seq, self.input_index, self.label = [], [], [], [], []
        for key in data:
            for one_traj in data[key]:
                self.input_index.append(one_traj[0])
                self.label.append(key)

                # Do not use merge window
                # traj = one_traj[1]
                # time = one_traj[2]
                # state = one_traj[3]

                # Use merge window
                traj = [one_traj[1][0]] + [one_traj[1][idx]
                                           for idx in range(1, len(one_traj[1])) if one_traj[1][idx] != one_traj[1][idx-1]]
                time = [one_traj[2][0]] + [one_traj[2][idx]
                                           for idx in range(1, len(one_traj[1])) if one_traj[1][idx] != one_traj[1][idx-1]]
                state = [one_traj[3][0]] + [one_traj[3][idx]
                                            for idx in range(1, len(one_traj[1])) if one_traj[1][idx] != one_traj[1][idx-1]]

                self.input_seq.append(traj)
                self.time_seq.append(time)
                self.state_seq.append(state)

    def __getitem__(self, index):
        return self.input_seq[index], self.time_seq[index], self.state_seq[index], self.input_index[index], self.label[index]

    def __len__(self):
        return len(self.input_seq)


def collate_fn(batch):
    """[summary]

    Args:
        batch ([type]): [description]

    Returns:
        [type]: [description]
    """
    traj_contents, time_contents, state_contents, indexs, labels = zip(*batch)
    max_len = max([len(content) for content in traj_contents])

    traj_contents = torch.LongTensor([content + [-1] * (max_len - len(content)) if len(
        content) < max_len else content for content in traj_contents])
    time_contents = torch.LongTensor([content + [124] * (max_len - len(
        content)) if len(content) < max_len else content for content in time_contents])
    state_contents = torch.LongTensor([content + [9] * (max_len - len(content)) if len(
        content) < max_len else content for content in state_contents])

    indexs = torch.LongTensor(indexs)
    labels = torch.LongTensor(labels)
    return traj_contents, time_contents, state_contents, indexs, labels


def get_dataloader(train, dataset, batch_size, sampler=None):
    """[summary]

    Args:
        train ([type]): [description]
        dataset ([type]): [description]
        batch_size ([type]): [description]
        sampler ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if train:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                shuffle=True, drop_last=False, collate_fn=collate_fn)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                sampler=sampler, drop_last=False, collate_fn=collate_fn)
    return dataloader


def get_dataset(test_nums, user_traj_train, user_traj_test):
    """[summary]

    Args:
        test_nums ([type]): [description]
        user_traj_train ([type]): [description]
        user_traj_test ([type]): [description]

    Returns:
        [type]: [description]
    """
    valid_size = 0.5
    indices = list(range(test_nums))
    # np.random.seed(555)  # # Fixed random seed(random seed 555 can be used to reproduce the results of this paper)
    np.random.shuffle(indices)
    split = int(np.floor(test_nums * valid_size))
    valid_idx, test_idx = indices[split:], indices[:split]
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_dataset = MolDataset(data=user_traj_train)
    test_dataset = MolDataset(data=user_traj_test)
    return train_dataset, test_dataset, valid_sampler, test_sampler


if __name__ == '__main__':
    from utils import load_data
    from datasets import get_dataloader
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    local_feature, local_adj, global_feature, global_adj, user_traj_train, user_traj_test, grid_nums, traj_nums, user_nums = load_data(
        dataset='geolife-walk-mini')
    local_feature, local_adj, global_feature, global_adj = local_feature.to(
        device), local_adj.to(device), global_feature.to(device), global_adj.to(device)
    loader = get_dataloader(train=True, data=user_traj_train)
    for idx, (input_seq, input_index, y_label) in enumerate(loader):
        input_seq, y_label = input_seq.to(device), y_label.to(device)
        print(idx)
        print(input_seq)
        print(input_index)
        print(y_label)
        break
