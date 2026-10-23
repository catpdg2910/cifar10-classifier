import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim=128, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0

        self.heads = heads
        self.head_dim = dim // heads

        # Ba phép chiếu học được
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape

        # [B, N, D] -> [B, heads, N, head_dim]
        q = self.q(x).reshape(
            B, N, self.heads, self.head_dim
        ).transpose(1, 2)
        k = self.k(x).reshape(
            B, N, self.heads, self.head_dim
        ).transpose(1, 2)
        v = self.v(x).reshape(
            B, N, self.heads, self.head_dim
        ).transpose(1, 2)

        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = self.attn_drop(scores.softmax(dim=-1))
        out = attention @ v

        # Ghép các head: [B, heads, N, head_dim] -> [B, N, D]
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out))


class EncoderBlock(nn.Module):
    def __init__(self, dim=128, heads=4, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = SelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-LayerNorm + residual connections
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SmallViT(nn.Module):
    def __init__(
        self, image_size=32, patch_size=4, dim=128,
        depth=4, heads=4, mlp_dim=512,
        num_classes=10, dropout=0.1
    ):
        super().__init__()
        assert image_size % patch_size == 0

        num_patches = (image_size // patch_size) ** 2

        # Conv không chồng lấn tương đương Linear trên từng patch
        self.patch_embedding = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, dim)
        )
        self.embedding_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            EncoderBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        # [B, 3, 32, 32] -> [B, 64, D]
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.embedding_drop(x + self.pos_embedding)
        x = self.blocks(x)

        # CLS token tổng hợp thông tin để phân loại
        return self.head(self.norm(x[:, 0]))
