import torch

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.head_dim = int(d_model / n_heads)

        # Parameters

        # attn
        self.Wqkv = torch.nn.Parameter(torch.randn(size=(3, self.n_heads, self.head_dim, self.d_model)))
        self.Wo = torch.nn.Parameter(torch.randn(size=(self.d_model, self.d_model)))

        # layer_norm
        self.scale = torch.nn.Parameter(torch.ones(size=(1,1,self.d_model)))
        self.bias = torch.nn.Parameter(torch.zeros(size=(1,1,self.d_model)))

        # mlp
        self.up_project = torch.nn.Parameter(torch.randn(size=(self.d_model, 4*self.d_model)))
        self.down_project = torch.nn.Parameter(torch.randn(size=(4* self.d_model, self.d_model)))


    def mha(self, x):
        batch, token, model_dim = x.shape
        qkv = torch.einsum('btd, snhd-> snbth', x, self.Wqkv)
        q = qkv[0,:]
        k = qkv[1,:]
        v = qkv[2,:]

        normalization_factor = 1 / torch.sqrt(torch.tensor(self.d_model))

        scores = normalization_factor* q @ k.mT

        weights = torch.softmax(scores, dim=-1) #need to understand the reduction here

        output = weights @ v # this matmul is off to me, I hesistated here

        return output.reshape(batch, token, model_dim)

    def mlp(self, x):
        gelu = torch.nn.GELU()
        x = x @ self.up_project
        x = gelu(x)
        x = x @ self.down_project 

        return x

    def layer_norm(self, x):
        eps = 1e-10
        mean = torch.mean(x, dim =-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)

        return self.scale * ((x - mean) / torch.sqrt(var + eps)) + self.bias

    def forward(self, data):
        ln_x1 = self.layer_norm(data)
        x = data + self.mha(ln_x1)
        ln_x2 = self.layer_norm(x)
        x = x + self.mlp(x)

        return x

B = 256
T = 1000
D = 64

X = torch.randn(size=(B, T, D))

block = TransformerBlock(d_model=D, n_heads=4) # head_dim = 16

print(block(X).shape)