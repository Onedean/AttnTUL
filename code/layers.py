import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """[gcn convolution layer]

    Args:
        Module ([type]): [description]
    """

    def __init__(self, in_features, out_features, bias=True):
        """[summary]

        Args:
            in_features ([int]): [input feature]
            out_features ([int]): [output feature]
            bias (bool, optional): [use bias or not]. Defaults to True.
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj):
        """[forward function]

        Args:
            feature ([torch.tensor]): [graph feature]
            adj ([torch.tensor]): [graph adj]

        Returns:
            [torch.tensor]: [convolution result]
        """
        support = torch.mm(feature, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class PositionalEncoding(nn.Module):
    """[position encoder]

    Args:
        nn ([type]): [description]
    """

    def __init__(self, d_model, dropout=0.1, max_len=200):
        """[summary]

        Args:
            d_model ([type]): [description]
            dropout (float, optional): [description]. Defaults to 0.1.
            max_len (int, optional): [description]. Defaults to 200.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos((position * div_term)[:, :-1]
                                ) if d_model % 2 else torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    """[summary]

    Args:
        seq_q ([type]): [description]
        seq_k ([type]): [description]

    Returns:
        [type]: [description]
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(-1).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """

    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        """[summary]

        Args:
            Q ([type]): [description]
            K ([type]): [description]
            V ([type]): [description]
            attn_mask ([type]): [description]

        Returns:
            [type]: [description]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        # Fills elements of self tensor with value where mask is one.
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """

    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model, self.d_k, self.d_v, self.n_heads = d_model, d_k, d_v, n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.scadot = ScaledDotProductAttention(d_k)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        """[summary]

        Args:
            Q ([type]): [description]
            K ([type]): [description]
            V ([type]): [description]
            attn_mask ([type]): [description]

        Returns:
            [type]: [description]
        """
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads,
                               self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context = self.scadot(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return self.norm(output + residual)


class PoswiseFeedForwardNet(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """

    def __init__(self, d_model, d_ff):
        """[summary]

        Args:
            d_model ([type]): [description]
            d_ff ([type]): [description]
        """
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """[summary]

        Args:
            inputs ([type]): [description]

        Returns:
            [type]: [description]
        """
        residual = inputs
        output = self.fc(inputs)
        return self.norm(output + residual)


class EncoderLayer(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """

    def __init__(self, d_model, d_k, d_v, d_ff, n_heads):
        """[summary]

        Args:
            d_model ([type]): [description]
            d_k ([type]): [description]
            d_v ([type]): [description]
            d_ff ([type]): [description]
            n_heads ([type]): [description]
        """
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """[summary]

        Args:
            enc_inputs ([type]): [description]
            enc_self_attn_mask ([type]): [description]

        Returns:
            [type]: [description]
        """
        enc_outputs = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs


class complexSparsemax(nn.Module):
    """[This is a complexSparsemax function of elastic attention.]

    Args:
        nn ([type]): [description]
    """

    def __init__(self, dim=None):
        """[Initialize sparsemax activation]

        Args:
            dim ([int], optional): [The dimension over which to apply the sparsemax function.]. Defaults to None.
        """
        super(complexSparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """[Forward function.]

        Args:
            input ([torch.Tensor]): [Input tensor. First dimension should be the batch size]

        Returns:
            [torch.Tensor]: [batch_size x number_of_logits]
        """
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1
        number_of_logits = input.size(dim)
        input = input - torch.max(input, dim=dim,
                                  keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1,
                             device=torch.device('cuda'), dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """[Backward function.]

        Args:
            grad_output ([type]): [description]

        Returns:
            [type]: [description]
        """
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / \
            torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input
