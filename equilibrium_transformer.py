import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm
from torchdeq.utils import mem_gc


class AttnInterface(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            cond=False
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        self.cond = cond

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, c, u=None):
        B, N, C = x.shape
        qkv = self.qkv(x)
        
        # Injection
        if self.cond:
            qkv = qkv + c
        if u is not None:
            qkv = qkv + u

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
    A Transformer block with equilibrium dynamics and additive attention injection.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cond=False, **block_kwargs):
        super().__init__()
        # Attention
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = AttnInterface(hidden_size, num_heads=num_heads, qkv_bias=True, cond=cond, **block_kwargs)
        
        # MLP
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        act = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=act, drop=0)
 
    def forward(self, x, c, u=None):
        x = x + self.attn(self.norm1(x), c, u)
        x = x + self.mlp(self.norm2(x))
        return x


class EquilibriumTransformer(nn.Module):
    """
    Implementation of the Equilibrium Transformer component.
    This transformer finds fixed points through deep equilibrium dynamics.
    """
    def __init__(
        self,
        hidden_size,
        deq_depth=3,
        num_heads=16,
        mlp_ratio=16.0,
        cond=False,
        mem_efficient=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.deq_depth = deq_depth
        self.mem_efficient = mem_efficient

        # DEQ blocks
        self.deq_blocks = nn.ModuleList([
            EquilibriumBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, cond=cond) 
            for _ in range(deq_depth)
        ])
        
        # Initialize weights
        self.initialize_weights()
        
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x, c=None, u_list=None, deq_solver=None):
        """
        Forward pass of Equilibrium Transformer.
        Args:
            x: Input tensor of shape (B, N, D)
            c: Conditional input (optional)
            u_list: List of injection vectors from injection transformer
            deq_solver: Deep equilibrium solver function
        """
        if deq_solver is None:
            raise ValueError("deq_solver must be provided")
            
        def func(z):
            for block, u in zip(self.deq_blocks, u_list):
                if self.mem_efficient:
                    z = mem_gc(block, (z, c, u))
                else:
                    z = block(z, c, u)
            return z
        
        # Initialize with random noise
        z = torch.randn_like(x)
        
        # Find fixed point
        z_out, info = deq_solver(func, z)
        
        if self.training:
            # Return all intermediate points for fixed point correction
            return z_out
        else:
            # Return only the final equilibrium point
            return z_out[-1] 