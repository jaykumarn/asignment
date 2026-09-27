import torch
import math
import torch.nn as nn
import torch.nn.functional as F

import tiktoken

class rmsnorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):

        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight



class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        """
        Args:
            dim (int): The embedding dimension per head (head_dim). Must be even.
            max_seq_len (int): Maximum expected sequence length to precompute cache.
            base (float): The geometric progression base for frequencies.
        """
        super().__init__()
        assert dim % 2 == 0, "Head dimension must be an even integer."
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 1. Generate cosine and sine frequency tensors (Precomputed Cache)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Shapes cached at: [1, 1, max_seq_len, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to split the last dimension in half and rotate."""
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies rotary transformations to Query and Key vectors while supporting 
        variable sequence lengths and preserving all tensor dimensions.

        Args:
            q (Tensor): Query matrix. Shape [batch_size, num_heads, seq_len, head_dim]
            k (Tensor): Key matrix. Shape [batch_size, num_heads, seq_len, head_dim]
            seq_len (int): Current batch sequence length.
            
        Returns:
            q_rotated, k_rotated (Tensor): Rotated tensors matching the exact input dimensions.
        """
        # 2. Support variable sequence lengths (Dynamic fallback or cache slice)
        if seq_len > self.max_seq_len:
            # Dynamically compute frequencies on-the-fly if batch exceeds max cache
            t = torch.arange(seq_len, device=q.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(q.device))
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()[None, None, :, :].to(q.dtype)
            sin = emb.sin()[None, None, :, :].to(q.dtype)
        else:
            # Slices cache dynamically to match variable batch seq_len
            cos = self.cos_cached[:, :, :seq_len, :].to(q.dtype)
            sin = self.sin_cached[:, :, :seq_len, :].to(q.dtype)
            
        # 3. Apply rotary transformations & 4. Preserve tensor dimensions
        # Mathematical formula: R(x) = x * cos + rotate_half(x) * sin
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rotated, k_rotated


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, num_query_heads: int, num_kv_heads: int,  max_seq_len: int = 4096):
        """
        Args:
            dim (int): The overall hidden/embedding dimension of the model.
            num_query_heads (int): Number of attention heads for the Query tensor.
            num_kv_heads (int): Number of attention heads for Key and Value tensors.
        """
        super().__init__()
        self.dim = dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_query_heads
        
        # Validation checks for GQA constraints
        assert dim % num_query_heads == 0, "dim must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        
        # Calculate how many Q heads share a single KV head group
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        
        # Requirement 1 & 2: Separate projections for Query and KV with configurable head counts
        self.wq = nn.Linear(dim, num_query_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        
        # Requirement 4: Final projection back to the original hidden dimension
        self.wo = nn.Linear(num_query_heads * self.head_dim, dim, bias=False)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeats KV heads to match the number of Query heads in each group.
        Input shape:  [batch_size, num_kv_heads, seq_len, head_dim]
        Output shape: [batch_size, num_query_heads, seq_len, head_dim]
        """
        if n_rep == 1:
            return x
            
        batch_size, num_kv_heads, seq_len, head_dim = x.shape
        
        # 1. Expand a new dimension for the repetitions within each group
        # Shape becomes: [batch_size, num_kv_heads, 1, seq_len, head_dim]
        x = x[:, :, None, :, :]
        
        # 2. Expand/repeat along the new dimension
        # Shape becomes: [batch_size, num_kv_heads, n_rep, seq_len, head_dim]
        x = x.expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
        
        # 3. Flatten the kv_heads and repetitions back into a single head dimension
        # Shape becomes: [batch_size, num_kv_heads * n_rep, seq_len, head_dim]
        return x.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Step 1: Separate Linear Projections
        xq = self.wq(x)  # [B, S, num_query_heads * head_dim]
        xk = self.wk(x)  # [B, S, num_kv_heads * head_dim]
        xv = self.wv(x)  # [B, S, num_kv_heads * head_dim]
        
        # Step 2: Reshape and transpose for multi-head layouts
        # Q shape:  [B, num_query_heads, S, head_dim]
        xq = xq.view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        # K shape:  [B, num_kv_heads, S, head_dim]
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # V shape:  [B, num_kv_heads, S, head_dim]
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # --- (If integrating RoPE, you would apply it to xq and xk here) ---
        
        # Step 3: Broadcast/Repeat KV heads to match the Query head group sizes
        # K and V scale up to shape: [B, num_query_heads, S, head_dim]
        xk = self._repeat_kv(xk, self.num_queries_per_kv)
        xv = self._repeat_kv(xv, self.num_queries_per_kv)
        
        # Step 4: Requirement 3 - Compute scaled dot-product attention
        # Scores shape: [B, num_query_heads, S, S]
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Softmax over the key sequence length
        probs = torch.softmax(scores, dim=-1)
        
        # Context calculation shape: [B, num_query_heads, S, head_dim]
        output = torch.matmul(probs, xv)
        
        # Step 5: Requirement 4 - Restore context and project back to original hidden dim
        # Permute/Flatten back to: [B, S, num_query_heads * head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Final linear output projection back to original dimension [B, S, dim]
        return self.wo(output)




class KVCache:
    def __init__(self):
        """Initializes an empty, reusable Key-Value cache structure."""
        self.k_cache = None
        self.v_cache = None

    @property
    def is_empty(self) -> bool:
        """Returns True if the cache has not been populated yet."""
        return self.k_cache is None or self.v_cache is None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new Key and Value pairs to the existing cache along the sequence dimension.
        
        Args:
            new_k (Tensor): New keys. Shape [batch_size, num_kv_heads, seq_len, head_dim]
            new_v (Tensor): New values. Shape [batch_size, num_kv_heads, seq_len, head_dim]
            
        Returns:
            k_full (Tensor), v_full (Tensor): The complete history of keys and values.
                                              Shape [batch_size, num_kv_heads, total_seq_len, head_dim]
        """
        # Requirement: Populate cache on the very first token/prefill pass
        if self.is_empty:
            self.k_cache = new_k
            self.v_cache = new_v
        else:
            # Requirement: Append new KV pairs along the sequence length dimension (dim=2)
            self.k_cache = torch.cat((self.k_cache, new_k), dim=2)
            self.v_cache = torch.cat((self.v_cache, new_v), dim=2)
            
        # Requirement: Return and reuse the fully combined cached tensors
        return self.k_cache, self.v_cache

    def get_seq_len(self) -> int:
        """Returns the current sequence length accumulated inside the cache."""
        if self.is_empty:
            return 0
        return self.k_cache.shape[2]

    def reset(self):
        """Requirement: Resets the cache state, freeing up GPU memory."""
        self.k_cache = None
        self.v_cache = None




class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        """
        Args:
            dim (int): The model hidden/embedding dimension (input and output depth).
            hidden_dim (int): The intermediate dimension inside the FFN layer.
                              Usually ~ (8/3) * dim for SwiGLU setups.
        """
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        # Requirement: Three linear layers configuration
        # w1 and w2 operate concurrently in parallel to form the gated branch
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Linear branch (gating filter)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)  # Gate activation branch
        
        # Requirement: Output projection back to the original model hidden dimension
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)  # Linear output map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input activations. Shape [batch_size, seq_len, dim]
        Returns:
            Tensor: Output activations matching input shape [batch_size, seq_len, dim]
        """
        # 1. Map into gate hidden dimensions
        gated_branch = self.w1(x)
        activation_branch = self.w2(x)
        
        # 2. Requirement: Swish (SiLU) activation function applied to one side
        swish_activated = F.silu(activation_branch)
        
        # 3. Requirement: Element-wise gating operation
        # Multiply the Swish output element-by-element with the linear branch
        gated_unit = swish_activated * gated_branch
        
        # 4. Requirement: Output projection back to hidden dimension
        return self.w3(gated_unit)

# --- 3. Final Integrated Transformer Decoder Block ---

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_query_heads: int, num_kv_heads: int, max_seq_len: int = 4096):
        """
        An integrated Pre-LN Transformer Decoder layer using GQA, RoPE, and SwiGLU.
        """
        super().__init__()
        # Pre-attention normalization
        self.attention_norm = rmsnorm(dim=dim)
        # GQA Module (containing internal RoPE positional tracking)
        self.attention = GroupedQueryAttention(
            dim=dim, 
            num_query_heads=num_query_heads, 
            num_kv_heads=num_kv_heads, 
            max_seq_len=max_seq_len
        )
        
        # Pre-FFN normalization
        self.ffn_norm = rmsnorm(dim=dim)
        # Standard hidden dimension size calculation for SwiGLU setups (~2/3 of 4d)
        hidden_dim = int(2 * (8 / 3) * dim // 2)
        self.feed_forward = SwiGLUFeedForward(dim=dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input structural activations. Shape [batch_size, seq_len, dim]
        """
        # Sub-Layer 1: Causal Attention + Pre-LN Residual Addition
        x = x + self.attention(self.attention_norm(x))
        
        # Sub-Layer 2: SwiGLU Feed-Forward + Pre-LN Residual Addition
        x = x + self.feed_forward(self.ffn_norm(x))
        
        return x


