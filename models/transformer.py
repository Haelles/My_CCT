import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, num_heads=2, embed_dim=128, attention_dropout=0.1, fc_dropout=0.1):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.sqrt_d = embed_dim ** -0.5
        self.single_dim = int(self.embed_dim / self.num_heads)

        self.linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.soft_max = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attention_dropout)

        self.fc = nn.Linear(embed_dim, embed_dim)
        self.fc_dropout = nn.Dropout(fc_dropout)

    def forward(self, x):  # b, n, d -> b, n, head, d/head -> b, head, n, d/head
        batch = x.size(0)
        n = x.size(1)
        q = self.linear_q(x).reshape(batch, n, self.num_heads, self.single_dim).transpose(1, 2)
        k = self.linear_k(x).reshape(batch, n, self.num_heads, self.single_dim).transpose(1, 2)
        v = self.linear_v(x).reshape(batch, n, self.num_heads, self.single_dim).transpose(1, 2)

        # score
        attn = self.soft_max(q @ k.transpose(2, 3) / self.sqrt_d)
        # 对score进行softmax
        attn = self.attn_dropout(attn)
        res = (attn @ v).transpose(1, 2).contiguous().reshape(batch, n, self.embed_dim)
        # 综合multi-head信息
        res = self.fc_dropout(self.fc(res))
        return res


class TransformEncoder(nn.Module):
    def __init__(self, embed_dim=128, ffn_dim=128,  dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):  # input: b, n, d
        super(TransformEncoder, self).__init__()
        self.pre_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(attention_dropout=attention_dropout, fc_dropout=dropout)
        self.norm_layer1 = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.activate = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)  # 根据author code添加上
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)  # 根据author code添加上
        # print("drop_path %f" % drop_path_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.drop_path(self.attn(self.pre_norm(x)))
        x = x + residual
        temp = self.norm_layer1(x)

        # FFN
        residual1 = temp
        temp = self.activate(self.linear1(temp))
        temp = self.linear2(self.dropout1(temp))
        res = residual1 + self.drop_path(self.dropout2(temp))

        return res


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
