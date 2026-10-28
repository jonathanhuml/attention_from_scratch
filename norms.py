import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.eps = 1e-8

        self.scale = torch.nn.Parameter(torch.ones(1, 1, self.d_model))
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, self.d_model))
    

    def forward(self, data):
        # mean and variance over vector dimensions d=-1
        mean = torch.mean(data, dim=-1, keepdim=True)
        var = torch.var(data, dim=-1, keepdim=True, unbiased=False) #subtract mean in the variance

        return self.scale * ((data - mean)/ torch.sqrt(var + self.eps)) + self.bias

X = torch.randn(size=(256, 128, 64))
LN = LayerNorm(d_model=X.shape[-1])
print(LN(X).shape)