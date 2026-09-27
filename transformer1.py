import torch
import math
import torch.nn as nn
import torch.nn.functional as F


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
        
        # Generate cosine and sine frequency tensors (Precomputed Cache)
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
        self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (Tensor): Query matrix. Shape [batch_size, num_heads, seq_len, head_dim]
            k (Tensor): Key matrix. Shape [batch_size, num_heads, seq_len, head_dim]
            start_pos (int): Starting sequence position index (used during KV cache decoding).
        """
        seq_len = q.shape[2]
        end_pos = start_pos + seq_len

        # Dynamic expansion or slicing from cache using absolute positions [start_pos:end_pos]
        if end_pos > self.max_seq_len:
            t = torch.arange(start_pos, end_pos, device=q.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(q.device))
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()[None, None, :, :].to(q.dtype)
            sin = emb.sin()[None, None, :, :].to(q.dtype)
        else:
            cos = self.cos_cached[:, :, start_pos:end_pos, :].to(q.device, dtype=q.dtype)
            sin = self.sin_cached[:, :, start_pos:end_pos, :].to(q.device, dtype=q.dtype)
            
        # Apply rotary transformations
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rotated, k_rotated


class KVCache:
    def __init__(self):
        self.k_cache = None
        self.v_cache = None

    @property
    def is_empty(self) -> bool:
        return self.k_cache is None or self.v_cache is None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_empty:
            self.k_cache = new_k
            self.v_cache = new_v
        else:
            self.k_cache = torch.cat((self.k_cache, new_k), dim=2)
            self.v_cache = torch.cat((self.v_cache, new_v), dim=2)
            
        return self.k_cache, self.v_cache

    def get_seq_len(self) -> int:
        if self.is_empty:
            return 0
        return self.k_cache.shape[2]

    def reset(self):
        self.k_cache = None
        self.v_cache = None


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, num_query_heads: int, num_kv_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_query_heads
        
        assert dim % num_query_heads == 0, "dim must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        
        self.wq = nn.Linear(dim, num_query_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_query_heads * self.head_dim, dim, bias=False)

        # -------------------------------------------------------------
        # Integrated RoPE Module (passing head_dim as dim parameter)
        # -------------------------------------------------------------
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        batch_size, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :]
        x = x.expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache: KVCache | None = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Step 1: Linear Projections
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # Step 2: Reshape to [B, Num_Heads, S, Head_Dim]
        xq = xq.view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # -------------------------------------------------------------
        # Step 3: Apply RoPE before KV Cache and Head Expansion
        # -------------------------------------------------------------
        start_pos = kv_cache.get_seq_len() if kv_cache is not None else 0
        xq, xk = self.rope(xq, xk, start_pos=start_pos)

        # Step 4: Update KV Cache (if provided)
        if kv_cache is not None:
            xk, xv = kv_cache.update(xk, xv)

        # Step 5: Repeat KV heads for GQA broadcast
        xk = self._repeat_kv(xk, self.num_queries_per_kv)
        xv = self._repeat_kv(xv, self.num_queries_per_kv)
        
        # Step 6: Scaled Dot-Product Attention
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Optional: Causal Masking
        if is_causal and seq_len > 1:
            kv_seq_len = xk.shape[2]
            mask = torch.full((seq_len, kv_seq_len), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1 + (kv_seq_len - seq_len))
            scores = scores + mask[None, None, :, :]

        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, xv)
        
        # Step 7: Restore Context Shape & Project Out
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w2(x)) * self.w1(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_query_heads: int, num_kv_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.attention_norm = rmsnorm(dim=dim)
        self.attention = GroupedQueryAttention(
            dim=dim, 
            num_query_heads=num_query_heads, 
            num_kv_heads=num_kv_heads, 
            max_seq_len=max_seq_len
        )
        self.ffn_norm = rmsnorm(dim=dim)
        hidden_dim = int(2 * (8 / 3) * dim // 2)
        self.feed_forward = SwiGLUFeedForward(dim=dim, hidden_dim=hidden_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache: KVCache | None = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        # Pre-LN Causal Attention with RoPE & Cache support
        x = x + self.attention(self.attention_norm(x), kv_cache=kv_cache, is_causal=is_causal)
        # Pre-LN SwiGLU FFN
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


import torch

def test_transformer_block():
    torch.manual_seed(42)

    # 1. Model Configuration
    batch_size = 2
    seq_len = 8
    dim = 64
    num_query_heads = 8
    num_kv_heads = 2
    max_seq_len = 128

    # Initialize the block
    block = TransformerBlock(
        dim=dim,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len
    )
    block.eval()  # Eval mode to disable dropout if added later

    print("--- Test 1: Full Sequence Forward Pass ---")
    x = torch.randn(batch_size, seq_len, dim)
    out_full = block(x, is_causal=True)

    assert out_full.shape == (batch_size, seq_len, dim), f"Expected shape {(batch_size, seq_len, dim)}, got {out_full.shape}"
    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {out_full.shape}")
    print("Pass 1 successful!\n")

    print("--- Test 2: Step-by-Step Generation with KV Cache ---")
    kv_cache = KVCache()
    cached_outputs = []

    # Process token by token
    for t in range(seq_len):
        x_token = x[:, t:t+1, :]  # Shape: [batch_size, 1, dim]
        out_step = block(x_token, kv_cache=kv_cache, is_causal=True)
        cached_outputs.append(out_step)

    out_cached = torch.cat(cached_outputs, dim=1)

    assert out_cached.shape == (batch_size, seq_len, dim), f"Expected shape {(batch_size, seq_len, dim)}, got {out_cached.shape}"
    assert kv_cache.get_seq_len() == seq_len, f"Cache sequence length mismatch: expected {seq_len}, got {kv_cache.get_seq_len()}"
    print(f"Aggregated Cached Output Shape: {out_cached.shape}")
    print(f"Final Cache Sequence Length:    {kv_cache.get_seq_len()}")
    print("Pass 2 successful!\n")

    print("--- Test 3: Numerical Equivalence Check ---")
    # Due to causal masking, full prompt output and step-by-step output must match
    max_diff = (out_full - out_cached).abs().max().item()
    print(f"Maximum difference between Full Pass and KV Cached Pass: {max_diff:.2e}")

    assert torch.allclose(out_full, out_cached, atol=1e-5), "Outputs between full forward pass and KV cached pass do not match!"
    print("Equivalence Test Passed successfully!\n")

if __name__ == "__main__":
    test_transformer_block()