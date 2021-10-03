import pickle
import random
from threading import local
import torch
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from torch._C import dtype
from tqdm import tqdm
from math import radians, cos, sin, asin, atan2, sqrt, degrees


def haversine_distance(lon1, lat1, lon2, lat2):
    """[This function is used to calculate the distance between two GPS points (unit: meter)]

    Args:
        lon1 ([float]): [longitude 1]
        lat1 ([float]): [latitude 1]
        lon2 ([float]): [longitude 2]
        lat2 ([float]): [latitude 2]

    Returns:
        [type]: [Distance between two points]
    """
    lon1, lat1, lon2, lat2 = map(
        radians, [lon1, lat1, lon2, lat2])  # Convert decimal degrees to radians
    c = 2 * asin(sqrt(sin((lat2 - lat1)/2)**2 + cos(lat1) *
                 cos(lat2) * sin((lon2 - lon1)/2)**2))  # Haversine formula
    r = 6371.393  # Average radius of the earth in kilometers
    return c * r * 1000


def conut_gird_num(tracks_data, grid_distance):
    """[This function is used to generate the number of lattice length and width according to the given lattice size]

    Args:
        tracks_data ([object]): [Original trajectory data]
        grid_distance ([int]): [Division distance]

    Returns:
        [type]: [description]
    """
    Lon1 = tracks_data['Lon'].min()
    Lat1 = tracks_data['Lat'].min()
    Lon2 = tracks_data['Lon'].max()
    Lat2 = tracks_data['Lat'].max()
    low = haversine_distance(Lon1, Lat1, Lon2, Lat1)
    high = haversine_distance(Lon1, Lat2, Lon2, Lat2)
    left = haversine_distance(Lon1, Lat1, Lon1, Lat2)
    right = haversine_distance(Lon2, Lat1, Lon2, Lat2)
    lon_grid_num = int((low + high) / 2 / grid_distance)
    lat_grid_num = int((left + right) / 2 / grid_distance)
    logger.info("After division, the whole map is:", lon_grid_num, '*',
          lat_grid_num, '=', lon_grid_num * lat_grid_num, 'grids')
    return lon_grid_num, lat_grid_num, Lon1, Lat1, Lon2, Lat2


def grid_process(tracks_data, grid_distance):
    """[This function is used to map each GPS point to a fixed grid]

    Args:
        tracks_data ([type]): [description]
        grid_distance ([type]): [description]

    Returns:
        [type]: [description]
    """
    lon_grid_num, lat_grid_num, Lon1, Lat1, Lon2, Lat2 = conut_gird_num(
        tracks_data, grid_distance)
    Lon_gap = (Lon2 - Lon1)/lon_grid_num
    Lat_gap = (Lat2 - Lat1)/lat_grid_num
    # Get the two-dimensional matrix coordinate index and convert it to one-dimensional ID
    tracks_data['grid_ID'] = tracks_data.apply(lambda x: int(
        (x['Lat']-Lat1)/Lat_gap) * lon_grid_num + int((x['Lon']-Lon1)/Lon_gap) + 1, axis=1)
    grid_list = sorted(set(tracks_data['grid_ID']))
    tracks_data['grid_ID'] = [grid_list.index(
        num) for num in tqdm(tracks_data['grid_ID'])]
    grid_list = sorted(set(tracks_data['grid_ID']))
    logger.info('After removing the invalid grid, there are', len(grid_list), 'grids')
    return tracks_data, grid_list


def generate_dataset(tracks_data, split_ratio):
    """[This function is used to generate data set, train set and test set]

    Args:
        tracks_data ([object]): [Trajectory data after discretization grid]
        split_ratio ([float]): [split ratio]

    Returns:
        [type]: [Track list, user list, data set, training set and test set, number of test sets]
    """
    user_list = tracks_data['ObjectID'].drop_duplicates().values.tolist()
    user_traj_dict = {key: [] for key in user_list}

    for user_id in tqdm(tracks_data['ObjectID'].drop_duplicates().values.tolist()):
        one_user_data = tracks_data.loc[tracks_data.ObjectID == user_id, :]
        for traj_id in one_user_data['TrajNumber'].drop_duplicates().values.tolist():
            # one_traj_data = one_user_data.loc[tracks_data.TrajNumber == traj_id, 'grid_ID'].drop_duplicates().values.tolist()
            one_traj_data = one_user_data.loc[tracks_data.TrajNumber ==
                                              traj_id, 'grid_ID'].values.tolist()
            one_time_data = one_user_data.loc[tracks_data.TrajNumber ==
                                              traj_id, 'time'].values.tolist()
            one_state_data = one_user_data.loc[tracks_data.TrajNumber ==
                                               traj_id, 'state'].values.tolist()
            user_traj_dict[user_id].append(
                (traj_id, one_traj_data, one_time_data, one_state_data))
    traj_list = list(range(traj_id+1))

    test_nums = 0
    user_traj_train, user_traj_test = {
        key: [] for key in user_list}, {key: [] for key in user_list}
    for key in user_traj_dict:
        traj_num = len(user_traj_dict[key])
        test_nums += traj_num - int(traj_num*split_ratio)
        for idx in list(range(traj_num))[:int(traj_num*split_ratio)]:
            user_traj_train[key].append(user_traj_dict[key][idx])
        for idx in list(range(traj_num))[int(traj_num*split_ratio):]:
            user_traj_test[key].append(user_traj_dict[key][idx])

    return traj_list, user_list, user_traj_dict, user_traj_train, user_traj_test, test_nums


def preprocess_adj(adj):
    """[A^~ = A + I]

    Args:
        adj ([type]): [adjacency matrix]

    Returns:
        [type]: [A^~]
    """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def normalize_adj(adj):
    """[Fourier transform]

    Args:
        adj ([type]): [adjacency matrix A^~]

    Returns:
        [type]: [Matrix after Fourier transform]
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    # D^-0.5AD^0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """[Convert the matrix into sparse matrix and save it]

    Args:
        sparse_mx ([type]): [description]

    Returns:
        [type]: [sparse matrix]
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def generate_graph(grid_list, traj_list, user_list, user_traj_dict, user_traj_train):
    """[This function is used to generate local graph, local graph node feature, global graph, global graph node feature]

    Args:
        grid_list ([list]): [grid list]
        traj_list ([list]): [trajectory list]
        user_list ([list]): [user list]
        user_traj_dict ([dict]): [all trajectory data]
        user_traj_train ([dict]): [trajectory data in training set]

    Returns:
        [type]: [local_feature, local_adj , local_feature, local_adj]
    """
    local_feature = np.eye(len(grid_list))

    local_graph = nx.Graph()
    local_graph.add_nodes_from(grid_list)
    local_edge_dict, local_edge_list = {}, []
    for key in user_traj_dict:
        for one_traj in user_traj_dict[key]:
            one_traj_list = one_traj[1]
            for idx in range(1, len(one_traj_list)):
                node1, node2 = sorted(
                    [one_traj_list[idx-1], one_traj_list[idx]])
                # if node1 != node2:
                if node1 != node2:
                    edge = str(node1) + ' ' + str(node2)
                    if edge not in local_edge_dict:
                        local_edge_dict[edge] = 1
                    else:
                        local_edge_dict[edge] += 1
    for key in local_edge_dict:
        local_edge_list.append(
            list(map(int, key.split()))+[local_edge_dict[key]])
    local_graph.add_weighted_edges_from(local_edge_list)
    local_adj = sparse_mx_to_torch_sparse_tensor(preprocess_adj(
        nx.to_scipy_sparse_matrix(local_graph, dtype=np.float)))

    global_feature = np.zeros((len(traj_list)+len(user_list), len(grid_list)))
    for key in user_traj_dict:
        sum_feature = np.zeros((1, len(grid_list)))
        for one_traj in user_traj_dict[key]:
            for idx in one_traj[1]:
                global_feature[one_traj[0], idx] += 1
            sum_feature += global_feature[one_traj[0]]
        global_feature[len(traj_list)+key] = sum_feature

    logger.info('Waiting to build global graph:')
    global_graph = nx.Graph()
    global_graph.add_nodes_from(
        traj_list + [len(traj_list)+idx for idx in user_list])
    global_edge_list, edge_weight_max = [], 0
    for node1 in tqdm(range(len(traj_list)-1)):
        node2_list = [idx for idx in range(node1+1, len(traj_list))]
        edge_weight_list = np.sum(np.minimum(
            global_feature[node1], global_feature[node2_list]), axis=1)
        edge_weight_max = max(edge_weight_list.max(), edge_weight_max)
        for node2, edge_weight in zip(node2_list, edge_weight_list):
            if edge_weight:
                global_edge_list.append([node1, node2, edge_weight])
    for key in user_traj_train:
        node1 = len(traj_list) + key
        for one_traj in user_traj_train[key]:
            node2 = one_traj[0]
            global_edge_list.append([node1, node2, edge_weight_max])
    global_graph.add_weighted_edges_from(global_edge_list)
    global_adj = sparse_mx_to_torch_sparse_tensor(preprocess_adj(
        nx.to_scipy_sparse_matrix(global_graph, dtype=np.float)))

    return torch.FloatTensor(local_feature), local_adj, torch.FloatTensor(global_feature), global_adj


def get_data_and_graph(raw_path, read_pkl, grid_size):
    """[Functions for processing data and generating local and global graphs]

    Args:
        raw_path ([str]): [Path of data file to be processed]
        read_pkl ([bool]): [If the value is false, the preprocessed data will be saved for direct use next time]
        grid_size ([type]): [Size of a single grid]

    Returns:
        [type]: [Processed trajectory data and graphs data]
    """
    grid_distance = grid_size
    split_ratio = 0.6

    tracks_data = pd.read_csv(raw_path, sep='\t')
    tracks_data, grid_list = grid_process(tracks_data, grid_distance)

    traj_list, user_list, user_traj_dict, user_traj_train, user_traj_test, test_nums = generate_dataset(
        tracks_data, split_ratio)
    local_feature, local_adj, global_feature, global_adj = generate_graph(
        grid_list, traj_list, user_list, user_traj_dict, user_traj_train)
    grid_nums, traj_nums, user_nums = len(
        grid_list), len(traj_list), len(user_list)

    if read_pkl == False:
        f = open(raw_path.replace('raw', 'process').replace(
            '.csv', '-'+str(grid_size)+'.pkl'), 'wb')
        pickle.dump((local_feature, local_adj, global_feature, global_adj, user_traj_train,
                    user_traj_test, grid_nums, traj_nums, user_nums, test_nums), f)
        f.close()
    return local_feature, local_adj, global_feature, global_adj, user_traj_train, user_traj_test, grid_nums, traj_nums, user_nums, test_nums
