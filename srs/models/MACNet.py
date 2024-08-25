import copy
import math
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from srs.layers.seframe import SEFrame
from srs.utils.data.collate import CollateFnContrativeLearning
from srs.utils.data.load import BatchSampler
from srs.utils.Dict import Dict
from srs.utils.prepare_batch import prepare_batch_factory_recursive
from srs.utils.data.transform import seq_to_weighted_graph

class AutoCorrAttention(nn.Module):
    """
    Implements the AutoCorrelation Attention Mechanism in a neural network.

    Args:
        hidden_dim (int): Dimensionality of the hidden layer.
        session_len (int): Length of the session.
        max_seq_length (int, optional): Maximum length of the sequence. Defaults to 50.
    """
    def __init__(self, hidden_dim, session_len, max_seq_length=50):
        super().__init__()
        self.attn_w0 = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.attn_w1 = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.attn_w2 = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.attn_w3 = nn.Parameter(torch.Tensor(session_len, hidden_dim))
        self.attn_bias = nn.Parameter(torch.Tensor(hidden_dim))
        
        # Parameter initialization
        init.normal_(self.attn_w0, mean=0, std=0.05)
        init.normal_(self.attn_w1, mean=0, std=0.05)
        init.normal_(self.attn_w2, mean=0, std=0.05)
        init.normal_(self.attn_w3, mean=0, std=0.05)
        init.constant_(self.attn_bias, 0)

        self.max_seq_length = max_seq_length
        self.complex_weight = nn.Parameter(
            torch.randn(1, 4, self.max_seq_length // 2 + 1, hidden_dim, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, q, k, v):
        """
        Forward pass for the AutoCorrelation Attention Mechanism.

        Args:
            q (Tensor): Query tensor.
            k (Tensor): Key tensor.
            v (Tensor): Value tensor.

        Returns:
            Tensor: Output tensor after applying the autocorrelation attention.
        """
        query = q.matmul(self.attn_w0)
        key = k.matmul(self.attn_w1) 
        value = v.matmul(self.attn_w2)
        
        query_fft = torch.fft.rfft(query, dim=2, norm='ortho')
        key_fft = torch.fft.rfft(key, dim=2, norm='ortho')

        # Autocorrelation attention
        res = query_fft * torch.conj(key_fft)
        weight = torch.view_as_complex(self.complex_weight)
        res *= weight
        corr = torch.fft.irfft(res, n=self.max_seq_length, dim=2, norm='ortho')

        alpha = torch.matmul(torch.relu(corr + self.attn_bias), self.attn_w3.t())
        alpha = F.softmax(alpha, dim=-2)

        x = torch.matmul(alpha.transpose(-1, -2), value)
        return x

class MACNet(nn.Module):
    """
    Implements the Multi-Head AutoCorrelation Network.

    Args:
        hidden_dim (int): Dimensionality of the hidden layer.
        session_len (int): The length of the session.
        num_heads (int): Number of heads in the multi-head attention mechanism.
        max_seq_length (int): Maximum sequence length.
    """
    def __init__(self, hidden_dim, session_len, num_heads=1, max_seq_length=50):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.d_k = hidden_dim // num_heads
        self.num_heads = num_heads

        self.attn = AutoCorrAttention(self.d_k, session_len, max_seq_length)
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(hidden_dim, hidden_dim)) for _ in range(num_heads)])
        self.intermediate_act_fn = gelu
        self.layer_norm = LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.5)

    def forward(self, query, key, value):
        """
        Forward pass for the MACNet.

        Args:
            query (Tensor): Query tensor.
            key (Tensor): Key tensor.
            value (Tensor): Value tensor.

        Returns:
            Tensor: The output tensor after applying the multi-head attention and subsequent operations.
        """
        nbatches = query.size(0)

        # Prepare queries, keys, and values for attention
        query, key, value = [
            x.view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2) for x in (query, key, value)
        ]

        # Apply auto-attention
        x = self.attn(query, key, value)

        # Concatenate and apply linear layer
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        x = self.linears[-1](x)

        # Apply dropout, activation function, and layer normalization
        x = self.dropout(x)
        x = self.intermediate_act_fn(x)
        x = self.layer_norm(x)

        return x

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SNARM(SEFrame):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        knowledge_graph,
        num_layers,
        batch_norm=True,
        feat_drop=0.0,
        **kwargs
    ):
        super().__init__(
            num_users,
            num_items,
            embedding_dim,
            knowledge_graph,
            num_layers,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            **kwargs,
        )
        self.pad_embedding = nn.Embedding(1, embedding_dim, max_norm=1)
        self.pad_indices = nn.Parameter(torch.arange(1, dtype=torch.long), requires_grad=False)
        self.pos_embedding = nn.Embedding(50, embedding_dim, max_norm=1)

        self.fc_i = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_u = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.fc_sr = nn.Linear(6 * embedding_dim, embedding_dim, bias=False)

        self.mul1 = MACNet(2*embedding_dim, 50, 4, max_seq_length=50)
        self.mul2 = MACNet(2*embedding_dim, 1, 4, max_seq_length=49) # use the last click as the q

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, inputs, extra_inputs=None):
        KG_embeddings = super().forward(extra_inputs)
        KG_embeddings["item"] = torch.cat([KG_embeddings["item"], self.pad_embedding(self.pad_indices)], dim=0)
        
        uids, padded_seqs, pos, g = inputs
        feat_i = KG_embeddings['item'][g.ndata['iid']]
        feat_u = KG_embeddings['user'][uids]

        last_nodes = g.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        ct_l = feat_i[last_nodes] 

        emb_seqs = KG_embeddings["item"][padded_seqs]
        pos_emb = self.pos_embedding(pos)
        feat = torch.cat([emb_seqs, pos_emb.unsqueeze(0).expand(emb_seqs.shape)], dim=-1)  # Combined sequence features: emb + pos
           
        attn_output1 = self.dropout1(self.mul1(self.dropout1(feat), feat, feat))
        # use the last click as the query
        attn_output2 = self.dropout2(self.mul2(attn_output1[:, -1, :], feat[:, :-1, :], feat[:, :-1, :]))
       
        sr = torch.cat([ct_l, feat_u, attn_output1[:,-1,:],attn_output2[:,-1,:]], dim=1)
        sequence_output = self.fc_sr(sr)
        logits = sequence_output @ self.item_embedding(self.item_indices).t()
        return sequence_output, logits

# Configuration dictionary
seq_to_graph_fns = [seq_to_weighted_graph]
config = Dict({
    'Model': SNARM,
    'CollateFn': CollateFnContrativeLearning,
    'seq_to_graph_fns': seq_to_graph_fns,
    'BatchSampler': BatchSampler,
    'prepare_batch_factory': prepare_batch_factory_recursive,
})


