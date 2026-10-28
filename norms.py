import torch

class LayerNorm(torch.nn.Module):
    def __init__(self, d_model):
        self.d_model = d_model

        self.scale = torch.ones(1, 1, self.d_model)
        self.bias = torch.zeros(1, 1, self.d_model)
    super().__init__()