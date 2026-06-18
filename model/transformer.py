import torch
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class regressoionHead(nn.Module):
    def __init__(self, d_embedding: int, dropout: float = 0.1):
        super().__init__()
        hidden1 = d_embedding // 2
        hidden2 = d_embedding // 4
        self.fc1 = nn.Linear(d_embedding, hidden1)
        self.norm1 = nn.LayerNorm(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.norm2 = nn.LayerNorm(hidden2)
        self.fc_out = nn.Linear(hidden2, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.gelu(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.gelu(self.norm2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc_out(x)


class TransformerPretrain(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1, proj_dim: int = 128,
                 pad_token_id: int = 0):
        super().__init__()
        self.model_type = 'Transformer'
        self.pad_token_id = pad_token_id
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )

        self.token_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, proj_dim)
        )

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor, return_tokens: bool = False,
                return_cls: bool = False, return_mean: bool = False):
        pad_mask = (src == self.pad_token_id)
        src = self.token_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=pad_mask)

        if return_mean:
            valid_mask = (~pad_mask).float().unsqueeze(-1)
            pooled = (output * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
            if return_tokens:
                token_embeds = self.token_proj(output[:, 1:, :])
                return pooled, token_embeds
            return pooled

        cls_embed = output[:, 0, :]

        if return_cls:
            if return_tokens:
                token_embeds = self.token_proj(output[:, 1:, :])
                return cls_embed, token_embeds
            return cls_embed

        cls_proj = self.proj_head(cls_embed)
        if return_tokens:
            token_embeds = self.token_proj(output[:, 1:, :])
            return cls_proj, token_embeds

        return cls_proj
