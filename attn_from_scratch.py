import torch 
import torch.nn 

embedding_dim = 64
batch_size = 256
n_tokens = 128

n_heads = 2
kqv_dim = 32 # = v_dim = 32 for now

assert n_heads * kqv_dim == embedding_dim

X = torch.rand(size=(batch_size, n_tokens, embedding_dim))
Wqkv = torch.nn.Parameter(torch.rand(size=(n_heads, 3, embedding_dim, kqv_dim)))

QKV = torch.einsum(
    "bte,hfed->hfbtd",
    X,
    Wqkv
)

scores = QKV[:,0,:] @ QKV[:,1,:].mT
variance_normalizing = kqv_dim ** -0.5


# add causal masking
output = torch.softmax(variance_normalizing* scores, dim=-1) @ QKV[:,2,:]

# concatenate heads 
#torch.Size([2, 256, 128, 32]) ---> torch.Size([256, 128, 64])
output = output.movedim(0,2).flatten(-2)

#linearly mix heads
Wo = torch.nn.Parameter(
    torch.rand(n_heads * kqv_dim, embedding_dim)
)

output = output @ Wo

#print(QKV.shape)
print(scores.shape)
print(output.shape)