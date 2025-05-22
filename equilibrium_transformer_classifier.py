import torch
import torch.nn as nn
import math
import numpy as np
from typing import Tuple

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 384,
        bias: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = (patch_size, patch_size)
        self.grid_size = img_size // patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, H/P * W/P, D)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    """Multi-head self attention."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divisible by num_heads {num_heads}'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        assert C == self.dim, f'Input dim {C} does not match layer dim {self.dim}'
        
        # Compute QKV with shape (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each has shape (B, num_heads, N, head_dim)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        return x

class GETBlock(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        # Ensure input is float32 for numerical stability
        x = x.to(dtype=torch.float32)
        
        # First normalization and attention
        x1 = self.norm1(x)
        x = x + self.attn(x1)
        
        # Second normalization and MLP
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Create 2D sine-cosine positional embeddings.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def anderson(f, x0, max_iter=50, m=5, lam=1e-4, threshold=50, eps=1e-5):
    """ Anderson acceleration for fixed point iteration. """
    bsz = x0.shape[0]
    
    # Make sure input is contiguous and on the correct device
    x0 = x0.contiguous()
    device = x0.device
    dtype = x0.dtype
    
    # Calculate flattened size
    flat_size = x0.numel() // bsz
    
    X = torch.zeros(bsz, m, flat_size, dtype=dtype, device=device)
    F = torch.zeros(bsz, m, flat_size, dtype=dtype, device=device)
    
    # Initial steps with proper reshaping and device placement
    X[:,0] = x0.reshape(bsz, -1)
    F[:,0] = f(x0).contiguous().reshape(bsz, -1)
    X[:,1] = F[:,0]
    F[:,1] = f(F[:,0].reshape_as(x0)).contiguous().reshape(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=dtype, device=device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=dtype, device=device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        
        # Add regularization for numerical stability
        reg_term = lam * torch.eye(n, dtype=dtype, device=device)[None]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + reg_term
        
        try:
            # Use more stable solver with better conditioning
            alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:, 1:n+1, 0]
        except RuntimeError:
            # If solver fails, fall back to simple iteration
            print(f"Warning: Solver failed at iteration {k}, falling back to simple iteration")
            X[:,k%m] = F[:,k%m-1]
            F[:,k%m] = f(X[:,k%m].reshape_as(x0)).contiguous().reshape(bsz, -1)
            continue
            
        # Update with computed coefficients
        X[:,k%m] = (alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].reshape_as(x0)).contiguous().reshape(bsz, -1)
        
        # Compute relative error
        res_k = (F[:,k%m] - X[:,k%m]).norm().item()
        denom = F[:,k%m].norm().item()
        res.append(res_k / (eps + denom))  # Avoid division by zero
        
        if (res[-1] < eps) or (k >= threshold):
            break
            
    return X[:,k%m].reshape_as(x0), res

class GETClassifier(nn.Module):
    """
    Classification model with only Equilibrium Transformer backbone.
    """
    def __init__(
        self,
        args,
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=1152,
        deq_depth=3,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.deq_depth = deq_depth
        self.num_classes = num_classes

        # Patch embedding
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # DEQ blocks
        self.deq_blocks = nn.ModuleList([
            GETBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(deq_depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

        self.initialize_weights()
        
        # DEQ parameters
        self.max_iter = getattr(args, 'f_thres', 30)
        self.stop_mode = getattr(args, 'stop_mode', 'rel')
        self.eps = getattr(args, 'f_eps', 1e-5)
        self.solver_eps = getattr(args, 'f_solver_eps', 1e-5)
        self.anderson_m = getattr(args, 'anderson_m', 5)

    def initialize_weights(self):
        # Initialize transformer layers
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

        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embedding
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize classification head
        nn.init.constant_(self.head.weight, 0)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        """
        Forward pass of GET Classifier.
        x: (B, C, H, W) tensor of spatial inputs (images)
        Returns: (B, num_classes) tensor of class logits
        """
        # Ensure input is on the correct device and dtype
        x = x.to(dtype=torch.float32)
        device = x.device
        
        # Patch embedding + positional embedding
        x = self.x_embedder(x)
        x = x + self.pos_embed.to(device=device, dtype=x.dtype)
        
        # DEQ forward function
        def func(z):
            for block in self.deq_blocks:
                z = block(z)
            return z
        
        # Initialize z with the input embedding
        z = x.clone().detach()
        z_out, res = anderson(
            func, z,
            max_iter=self.max_iter,
            m=self.anderson_m,
            eps=self.eps,
            threshold=self.max_iter
        )

        # Apply final normalization and classification head
        if self.training:
            # For fixed point correction, return logits for all iterations
            return [self.head(self.norm(z_out[:, 0]))]
        else:
            # For inference, return logits from final iteration
            return self.head(self.norm(z_out[:, 0])) 