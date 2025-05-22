import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Mlp
from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm
from torchdeq.utils import mem_gc


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return: [grid_size*grid_size, embed_dim] positional embeddings
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h)
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = pos.unsqueeze(-1) * omega.unsqueeze(0)  # (M, D/2)

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class AttnInterface(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EquilibriumBlock(nn.Module):
    """
    A Transformer block with equilibrium dynamics.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        # Attention
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = AttnInterface(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        # MLP
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)
 
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class EquilibriumClassifier(nn.Module):
    """
    Equilibrium Transformer for image classification.
    Specifically adapted for CIFAR dataset.
    """
    def __init__(
        self,
        img_size=32,           # CIFAR image size
        patch_size=4,          # Patch size for tokenization
        in_channels=3,         # RGB images
        num_classes=10,        # CIFAR-10 classes
        hidden_size=384,       # Embedding dimension
        deq_depth=3,          # Number of DEQ blocks
        num_heads=6,          # Number of attention heads
        mlp_ratio=4.0,        # MLP expansion ratio
        mem_efficient=False,   # Memory efficient computation
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.deq_depth = deq_depth
        self.mem_efficient = mem_efficient
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_channels, 
            embed_dim=hidden_size
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Add [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_size),  # +1 for [CLS] token
            requires_grad=False
        )

        # DEQ blocks
        self.deq_blocks = nn.ModuleList([
            EquilibriumBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) 
            for _ in range(deq_depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize [CLS] token
        nn.init.normal_(self.cls_token, std=0.02)

        # Initialize position embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5)
        )
        pos_embed = torch.cat([torch.zeros(1, self.pos_embed.shape[-1]), pos_embed])
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

    def forward(self, x, deq_solver=None):
        """
        Forward pass for classification.
        Args:
            x: Input images (B, C, H, W)
            deq_solver: Deep equilibrium solver function
        Returns:
            logits: Classification logits (B, num_classes)
        """
        if deq_solver is None:
            raise ValueError("deq_solver must be provided")

        # Patch embedding
        x = self.patch_embed(x)
        
        # Add [CLS] token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        def func(z):
            for block in self.deq_blocks:
                if self.mem_efficient:
                    z = mem_gc(block, (z,))
                else:
                    z = block(z)
            return z
        
        # Find fixed point
        z = torch.randn_like(x)
        z_out, _ = deq_solver(func, z)
        
        # Use the final equilibrium point
        if self.training:
            x = z_out[-1]
        else:
            x = z_out[-1]
            
        # Classification from [CLS] token
        x = self.norm(x[:, 0])
        x = self.head(x)
        
        return x 