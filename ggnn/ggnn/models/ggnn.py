import torch
from torch import nn
from torch import sparse
from torch.nn import functional as F


class GGNN(nn.Module):
    def __init__(self, n_tokens, n_types, n_edges, node_dim, token_dim, type_dim, annotation_dim, message_dim, n_steps):
        super(GGNN, self).__init__()
        self.n_steps = n_steps
        self.n_edges = n_edges
        self.node_dim = node_dim
        self.message_dim = message_dim
        self.message_generator = nn.Linear(node_dim, message_dim * n_edges)
        self.state_generator = nn.Sequential(
            nn.Linear(type_dim + token_dim, annotation_dim),
            nn.ConstantPad1d((0, node_dim - annotation_dim), 0))
        self.tokens = nn.EmbeddingBag(n_tokens, token_dim, mode='sum')
        self.types = nn.Embedding(n_types, type_dim)
        self.updater = nn.GRUCell(input_size=message_dim, hidden_size=node_dim)

    def forward(self, var_type, node_tokens, mask, adjacency_matrix):
        tokens = self.tokens(node_tokens, per_sample_weights=mask)
        types = self.types(var_type)
        state = self.state_generator(torch.cat([tokens, types], 1))
        divider = torch.clamp_min_(sparse.sum(adjacency_matrix, dim=1).to_dense(), 1).view(-1, 1)
        for j in range(self.n_steps):
            messages_out = self.message_generator(state).view((-1, self.message_dim))
            messages_in = sparse.mm(adjacency_matrix, messages_out) / divider
            state = self.updater(messages_in, state)
        return state


class MetricsPredictor(nn.Module):
    def __init__(self, n_tokens, n_types, n_edges, node_dim, token_dim, type_dim,
                 annotation_dim, message_dim, n_steps, n_metrics):
        super(MetricsPredictor, self).__init__()
        self.ggnn = GGNN(n_tokens, n_types, n_edges, node_dim, token_dim, type_dim,
                         annotation_dim, message_dim, n_steps)
        self.attention = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(node_dim, 1))
        self.predictor = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(node_dim, n_metrics))

    def forward(self, var_type, node_tokens, mask, adjacency_matrix, lens):
        states = self.ggnn(var_type, node_tokens, mask, adjacency_matrix)
        data = torch.nn.utils.rnn.pad_sequence(torch.split(states, lens.tolist()), batch_first=True)
        weight = F.softmax(self.attention(data), dim=1)
        result = torch.sum(torch.mul(data, weight), dim=1)
        return self.predictor(result), weight