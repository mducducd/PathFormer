import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
from typing import Optional, Any
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))

class CoordAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # MLP for continuous coordinate-based relative positional bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads)
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, coords, mask=None):
        """
        Args:
            x: (B, N, C) - input features
            coords: (B, N, 2) - real coordinates (e.g., WSI patch centers)
            mask: Optional attention mask (B, N, N)
        Returns:
            Output: (B, N, C)
        """
        B, N, C = x.shape
        # Compute QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # Scaled dot-product attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, num_heads, N, N)

        # Coordinate difference and bias computation
        rel_coords = coords[:, :, None, :] - coords[:, None, :, :]  # (B, N, N, 2)
        rel_coords = rel_coords / (rel_coords.norm(dim=-1, keepdim=True) + 1e-6)  # normalize direction
        bias = self.cpb_mlp(rel_coords)  # (B, N, N, num_heads)
        bias = bias.permute(0, 3, 1, 2)  # (B, num_heads, N, N)

        attn = attn + bias

        # Optional attention mask
        if mask is not None:
            attn = attn + mask.unsqueeze(1)  # (B, 1, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Apply attention to values
        out = (attn @ v)  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
# x = torch.randn(32, 128, 512)    
# coords = torch.randn(32, 128, 2)        
# attn = CoordAttention(dim=512, num_heads=8)
# output = attn(x, coords)
# print( output.shape)
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))

# class CrossAttention(nn.Module):
#     def __init__(self, d_model=512, n_heads=8):
#         super().__init__()
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, d_model)
#         )
#         self.layernorm1 = nn.LayerNorm(d_model)
#         self.layernorm2 = nn.LayerNorm(d_model)
        
#     def forward(self, query, key, value):
#         # Cross-attention
#         attn_output, _ = self.multihead_attn(query, key, value)
#         query = self.layernorm1(query + attn_output)
        
#         # Feed-forward
#         ffn_output = self.ffn(query)
#         query = self.layernorm2(query + ffn_output)
#         return query

# class TransformerEncoderLayer(nn.modules.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
#         super(TransformerEncoderLayer, self).__init__()
#         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
#         self.linear1 = Linear(d_model, dim_feedforward)
#         self.dropout = Dropout(dropout)
#         self.linear2 = Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
#         self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
#         self.dropout1 = Dropout(dropout)
#         self.dropout2 = Dropout(dropout)

#         self.activation = _get_activation_fn(activation)

#     def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
#                 src_key_padding_mask: Optional[Tensor] = None,
#                 **kwargs) -> Tensor:
#         # src: (seq_len, batch_size, d_model)
#         src2 = self.self_attn(src, src, src, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[0]
#         src = self.norm1(src + self.dropout1(src2))
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = self.norm2(src + self.dropout2(src2))
#         return src

# class TransformerEncoderClassifier(nn.Module):
#     """
#     Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
#     softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
#     """

#     def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, n_classes,
#                  dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
#         super(TransformerEncoderClassifier, self).__init__()

#         self.max_len = max_len
#         self.d_model = d_model
#         self.n_heads = n_heads

#         self.project_inp = nn.Linear(feat_dim, d_model)
#         self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)


#         encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

#         self.act = _get_activation_fn(activation)

#         self.dropout1 = nn.Dropout(dropout)

#         self.feat_dim = feat_dim
#         self.n_classes = n_classes
#         self.output_layer = self.build_output_module(d_model, max_len, n_classes)

#     def build_output_module(self, d_model, max_len, n_classes):
#         output_layer = nn.Linear(d_model * max_len, n_classes)
#         # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
#         # add F.log_softmax and use NLLoss
#         return output_layer

#     def forward(self, X, coords, padding_masks):
#         """
#         Args:
#             X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
#             padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
#         Returns:
#             output: (batch_size, n_classes)
#         """

#         # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
#         inp = X.permute(1, 0, 2)
#         inp = self.project_inp(inp) * math.sqrt(
#             self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
#         inp = self.pos_enc(inp)  # add positional encoding
#         # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
#         output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
#         output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
#         output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
#         output = self.dropout1(output)

#         # Output
#         output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
#         output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
#         output = self.output_layer(output)  # (batch_size, n_classes)

#         return output
    
# x = torch.randn(4, 128, 1024)              # features
# coords = torch.randn(4, 128, 2)         # WSI (x, y) positions
# mask = torch.randint(0, 2, (4, 128), dtype=torch.bool)
# model = TransformerEncoderClassifier(feat_dim=1024, 
#                                     max_len=128, 
#                                     d_model=512, 
#                                     n_heads=8, 
#                                     num_layers=2, 
#                                     dim_feedforward=2048,
#                                     n_classes=2)
# output = model(x, mask)
# print( output.shape)

class CTransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = CoordAttention(dim=512, num_heads=8)

    def forward(self, x, coords):
        x = x + self.attn(self.norm(x), coords)
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class CTransformer(nn.Module):
    def __init__(self, n_classes):
        super(CTransformer, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = CTransLayer(dim=512)
        self.layer2 = CTransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, h, coords, *args, **kwargs):

        h = self._fc1(h) #[B, n, 512]

        # pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        # Pad coords similarly?
        coords = torch.cat([coords, coords[:, :add_length, :]], dim=1)

        # cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # Add the [CLS] token coordinates (zero predefined)
        cls_coords = torch.zeros(B, 1, 2).cuda()
        coords = torch.cat((cls_coords, coords), dim=1)

        # Translayer x1
        h = self.layer1(h, coords) #[B, N, 512]

        # # PPEG
        # h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        # Translayer x2
        h = self.layer2(h, coords) #[B, N, 512]

        # cls_token
        h = self.norm(h)[:,0]

        # predict
        logits = self._fc2(h) # [B, n_classes]
        return logits