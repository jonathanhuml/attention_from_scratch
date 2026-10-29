import torch
from einops import rearrange

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # Layers
        self.norm1 = RMSNorm(d_model=self.d_model)
        self.mha = MHA(d_model=self.d_model, n_heads=self.n_heads)
        self.norm2 = RMSNorm(d_model=self.d_model)
        self.ffn = FFN(d_model=self.d_model)

    def forward(self, x):
        y = self.mha(self.norm1(x))
        x = x + y
        y = self.ffn(self.norm2(x))
        x = x + y
        return x



class FFN(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model

        # check initialization here later...
        self.up_project = torch.nn.Linear(in_features=self.d_model, out_features=4*self.d_model)
        self.sigma = torch.nn.GELU()
        self.down_project = torch.nn.Linear(in_features=4*self.d_model, out_features=self.d_model)

    def forward(self, x):
        x = self.up_project(x)
        x = self.sigma(x)
        x = self.down_project(x)
        return x


class MHA(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.head_dim = int(self.d_model / self.n_heads)

        # fix initialization here...
        self.WqWkWv = torch.nn.Parameter(torch.randn(size=(3, self.n_heads, self.d_model, self.head_dim)))
        self.Wo = torch.nn.Linear(in_features=self.d_model, out_features=self.d_model)


    def forward(self, x):
        # project input
        # s = 3 for set of W_q, W_k, W_v
        # n = number of heads
        # h = head_dim
        qkv = torch.einsum(' btd, sndh -> snbth', x, self.WqWkWv)
        q = qkv[0,:]
        k = qkv[1,:]
        v = qkv[2,:]

        normalization_factor = 1 / torch.sqrt(torch.tensor(self.head_dim))
        scores = normalization_factor * q @ k.mT
        regularized_scores = torch.softmax(scores, dim=-1)

        output = regularized_scores @ v
        output = rearrange(output, 'n b t h -> b t (h n)')
        linear_mix = self.Wo(output)
        return linear_mix


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.scale = torch.nn.Parameter(torch.ones(size=(1, 1, self.d_model)))

    def forward(self, x):
        # (B, T, D)
        # sqrt(mean(squared values))
        eps = 1e-8
        squared = torch.square(x)
        mean_squared = torch.mean(squared, dim=-1, keepdim=True)
        return self.scale * x / (torch.sqrt(mean_squared + eps))
B = 64
T = 100
D = 32
X = torch.randn(size=(B, T, D))
norm = RMSNorm(d_model=D)
print(norm(X).shape)

network = FFN(d_model=D)
print(network(X).shape)

mha = MHA(d_model=D, n_heads=2)
print(mha(X).shape)

block = TransformerBlock(d_model=D, n_heads=2)
print(block(X).shape)