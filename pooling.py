import torch
import torch.nn as nn
import torch.nn.functional as F


class TP(nn.Module):
    """
    temporal pooling
    """
    def __init__(self):
        super().__init__()

    def forward(self, frame_features):
        x = frame_features.squeeze(-2)
        pooling_output = torch.mean(x, dim=-1)
        return pooling_output


class TSP(nn.Module):
    """
    temporal statistical pooling
    """
    def __init__(self,
                 eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, frame_features):
        # input: batch_size, num_channels, 1, num_steps
        x = frame_features.squeeze(-2)
        mean = torch.mean(x, dim=-1)
        std = torch.std(x, dim=-1)
        pooling_output = torch.cat((mean, std), dim=-1)
        return pooling_output


class SMHAP(nn.Module):
    """
    self multi-head attention pooling
    paper: Self Multi-Head Attention for Speaker Recognition
    """
    def __init__(self,
                 in_dim,
                 n_heads):
        super(SMHAP, self).__init__()
        assert in_dim % n_heads == 0, print("{} % {} is not equal to zero.")
        self.head_dim = int(in_dim // n_heads)
        self.num_heads = n_heads
        self.u = nn.Parameter(torch.randn(size=(self.num_heads, self.head_dim)),
                              requires_grad=True)
        nn.init.xavier_uniform_(self.u)

    def forward(self, batch: torch.FloatTensor):
        batch_size, in_dim, _, time_steps = batch.size()
        x = batch.view(batch_size, self.num_heads, self.head_dim, time_steps)
        w = torch.einsum("ijkl,jk->ijl", x, self.u)
        normed_w = F.softmax(w, dim=-1)
        out1 = torch.einsum("ijkl,ijl->ijk", x, normed_w)
        out2 = out1.view(batch_size, self.num_heads * self.head_dim)
        return out2


class SASFunction(nn.Module):
    def __init__(self, n_dim, n_head, T, activation='ReLU'):
        super().__init__()
        self.n_head = n_head
        self.n_dim = n_dim
        self.fc = nn.Linear(n_dim, n_dim * n_head)
        self.V = nn.Parameter(torch.randn(n_head, n_dim), requires_grad=True)
        self.b = nn.Parameter(torch.randn(n_head), requires_grad=True)
        self.T = T
        if activation == 'ReLU':
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.xavier_uniform_(self.V)
        nn.init.constant_(self.b, 0.0)

    def forward(self, x):
        # x : B, N, T
        B, _, T = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = self.act(self.fc(x))
        x = x.view(B, T, self.n_head, -1)
        x = torch.einsum('ijkl, kl -> ijk', x, self.V) + self.b
        x = F.softmax(x / self.T.view(1, 1, -1), dim=1)
        return x.permute(0, 2, 1).contiguous()  # B, n_head, T


class MRMHAPooling(nn.Module):
    def __init__(self, n_dim, n_head):
        super().__init__()
        T = torch.FloatTensor([max(1, int((i-1)/2)*5) for i in range(1, n_head+1)])
        T = T.cuda()
        T.requires_grad = False
        self.sasfunc = SASFunction(n_dim, n_head, activation='ReLU', T=T)
        self.n_dim = n_dim
        self.n_head = n_head

    def forward(self, x):
        # x : B, N, 1, T
        x = x.squeeze(dim=-2)
        B, N, T = x.shape
        alpha = self.sasfunc(x).unsqueeze(2) 
        x = x.unsqueeze(1)
        e = torch.sum(x * alpha, dim=-1)
        e = e.view(B, -1)
        return e

