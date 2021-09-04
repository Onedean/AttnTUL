import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution
from layers import PositionalEncoding, get_attn_pad_mask, EncoderLayer, complexSparsemax


class GcnNet(nn.Module):
    """[Graph convolution network]

    Args:
        nn ([type]): [description]
    """

    def __init__(self, nfeat, nhid1, nhid2, dropout):
        """[Initialization function]

        Args:
            nfeat ([int]): [node feature dim]
            nhid1 ([int]): [hidden1 dim]
            nhid2 ([int]): [hidden2 dim]
            dropout ([float]): [dropout value]
        """
        super(GcnNet, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, feature, adj):
        """[forward function]

        Args:
            feature ([type]): [description]
            adj ([type]): [description]

        Returns:
            [type]: [description]
        """
        x = torch.relu(self.gc1(feature, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return x


class EncoderAttentionBlock(nn.Module):
    """[self-attention encoder block]

    Args:
        nn ([type]): [description]
    """

    def __init__(self, d_model, d_k, d_v, d_ff, n_heads, n_layers):
        super(EncoderAttentionBlock, self).__init__()
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_k, d_v, d_ff, n_heads) for _ in range(n_layers)])

    def forward(self, input_seq, input_seq_emb):
        """[forward function]

        Args:
            input_seq ([type]): [description]
            input_seq_emb ([type]): [description]

        Returns:
            [type]: [description]
        """
        enc_outputs = self.pos_emb(
            input_seq_emb.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(input_seq, input_seq)
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)
        return enc_outputs


class ElasticAttentionBlock(nn.Module):
    """[Elastic attention block]

    Args:
        nn ([type]): [description]
    """

    def __init__(self, Attn_Strategy, Softmax_Strategy):
        """[summary]

        Args:
            Attn_Strategy ([str]): [What attention strategies is used]
            Softmax_Strategy ([str]): [What normalization function strategy is used]
        """
        super(ElasticAttentionBlock, self).__init__()
        self.attn_strategy = Attn_Strategy
        self.softmax_strategy = Softmax_Strategy
        self.normSoftmax = nn.Softmax(dim=1)
        # self.simpleSoftmax = simpleSparsemax(dim=1)
        self.complexSoftmax = complexSparsemax(dim=1)

    def forward(self, querry_emb, traj_emb):
        """[Elastic attention calculation]

        Args:
            querry_emb ([torch.tensor]): [Querry]
            traj_emb ([torch.tensor]): [Key/Value]

        Returns:
            [torch.tensor]: [context]
        """
        if self.attn_strategy == 'cos':
            score = torch.matmul(querry_emb, traj_emb.T) / torch.matmul(torch.sqrt(torch.sum(querry_emb *
                                                                                             querry_emb, dim=1)).unsqueeze(1), torch.sqrt(torch.sum(traj_emb * traj_emb, dim=1)).unsqueeze(0))
        elif self.attn_strategy == 'dot':
            score = torch.matmul(querry_emb, traj_emb.T)

        if self.softmax_strategy == 'complex':
            atten = self.complexSoftmax(score)
        elif self.softmax_strategy == 'simple':
            pass  # todo
        elif self.softmax_strategy == 'norm':
            atten = self.normSoftmax(score)

        context = torch.matmul(atten, traj_emb)
        return context


class Embedding(nn.Module):
    """[Fusion auxiliary information]

    Args:
        nn ([type]): [description]
    """

    def __init__(self, d_model):
        """[summary]

        Args:
            d_model ([int]): [embedding dim]
        """
        super(Embedding, self).__init__()
        self.timeEmb = nn.Embedding(125, 32, padding_idx=124)  # time Embedding
        self.stateEmb = nn.Embedding(10, 32, padding_idx=9)  # state Embedding
        self.fc = nn.Linear(d_model + 64, d_model)
        self.tanh = nn.Tanh()

    def forward(self, input_seq_emb, time_seq, state_seq):
        """[forward function]

        Args:
            input_seq_emb ([torch.tensor]): [traj sequence]
            time_seq ([torch.tensor]): [time sequence]
            state_seq ([torch.tensor]): [state sequence]

        Returns:
            [torch.tensor]: [fusion embedding]
        """
        embedding = self.tanh(self.fc(torch.cat(
            [input_seq_emb, self.timeEmb(time_seq), self.stateEmb(state_seq)], dim=-1)))
        return embedding


class MolNet(nn.Module):
    """[stan's net]

    Args:
        nn ([type]): [description]
    """

    def __init__(self, Attn_Strategy, Softmax_Strategy, Pool_Strategy, d_model, d_k, d_v, d_ff, n_heads, n_layers, user_nums):
        """[summary]

        Args:
            Attn_Strategy ([str]): [Which attn strategy is used]
            Softmax_Strategy ([str]): [Which softmax strategy is used]
            Pool_Strategy ([str]): [Which pooling strategy is used]
            d_model ([int]): [embedding]
            d_k ([int]): [key dim]
            d_v ([int]): [value dim]
            d_ff ([int]): [ffn dim]
            n_heads ([int]): [Number of heads]
            n_layers ([int]): [Number of self attention layers]
            user_nums ([int]): [Number of users]
        """
        super(MolNet, self).__init__()
        self.pool_strategy = Pool_Strategy

        self.embedding = Embedding(d_model)
        self.LocalEncoderNet = EncoderAttentionBlock(
            d_model, d_k, d_v, d_ff, n_heads, n_layers)
        self.GlobalAttentionNet = ElasticAttentionBlock(
            Attn_Strategy, Softmax_Strategy)
        self.classifier = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, user_nums)
        )

    def forward(self, grid_emb, traj_emb, input_seq, time_seq, state_seq, input_index):
        """[forward function]

        Args:
            grid_emb ([torch.tensor]): [grid embedding lookup table]
            traj_emb ([torch.tensor]): [traj sequence]
            input_seq ([torch.tensor]): [the length of sequence]
            time_seq ([torch.tensor]): [time sequence]
            state_seq ([torch.tensor]): [state sequence]
            input_index ([list]): [Number corresponding to the trajectory]

        Returns:
            [torch.tensor]: [the result of trajectories]
        """
        input_seq_onehot = torch.zeros(
            input_seq.shape[0], input_seq.shape[1], grid_emb.shape[0]).cuda()
        for idx, one_input in enumerate(input_seq):
            for idy, index in enumerate(one_input[one_input != -1]):
                input_seq_onehot[idx, idy, index] = 1
        input_seq_emb = torch.matmul(input_seq_onehot, grid_emb)

        input_seq_emb = self.embedding(input_seq_emb, time_seq, state_seq)

        input_seq_emb = self.LocalEncoderNet(input_seq, input_seq_emb)

        if self.pool_strategy == 'max':
            input_seq_emb, _ = input_seq_emb.max(dim=1)
        else:
            input_seq_emb = input_seq_emb.mean(dim=1)

        querry_emb = traj_emb[input_index]
        context = self.GlobalAttentionNet(querry_emb, traj_emb)

        # context = self.GlobalAttentionNet(input_seq_emb, traj_emb)

        output = torch.cat([input_seq_emb, context], dim=-1)
        output = self.classifier(output)

        return F.log_softmax(output, dim=-1)
