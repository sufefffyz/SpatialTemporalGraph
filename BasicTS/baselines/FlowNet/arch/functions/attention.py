import torch as th
from torch.nn import functional as F

"""
    Implementation of RoPE and attention mechanism. 
    Code copied from https://blog.csdn.net/weixin_43646592/article/details/130924280.
"""


def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device):
    # (max_len, 1)
    position = th.arange(0, max_len, dtype=th.float).unsqueeze(-1)
    # (output_dim//2)
    ids = th.arange(0, output_dim // 2, dtype=th.float)
    theta = th.pow(10000, -2 * ids / output_dim)

    # (max_len, output_dim//2)
    embeddings = position * theta
    # (max_len, output_dim//2, 2)
    embeddings = th.stack([th.sin(embeddings), th.cos(embeddings)], dim=-1)
    # (bs, head, max_len, output_dim//2, 2)
    embeddings = embeddings.repeat(
        (batch_size, nums_head, *([1] * len(embeddings.shape)))
    )
    # (bs, head, max_len, output_dim)
    embeddings = th.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings


def RoPE(q, k):
    # q,k: (bs, head, max_len, output_dim)
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[3]

    # (bs, head, max_len, output_dim)
    pos_emb = sinusoidal_position_embedding(
        batch_size, nums_head, max_len, output_dim, q.device
    )
    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

    # q,k: (bs, head, max_len, output_dim)
    q2 = th.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)
    q = q * cos_pos + q2 * sin_pos
    k2 = th.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    k = k * cos_pos + k2 * sin_pos
    return q, k


def attention(q, k, v, use_RoPE=False, is_casual=False):
    # q.shape: (bs, head, seq_len, dk)
    # k.shape: (bs, head, seq_len, dk)
    # v.shape: (bs, head, seq_len, dk)

    if use_RoPE:
        q, k = RoPE(q, k)

    d_k = k.size()[-1]
    att_logits = th.matmul(q, k.transpose(-2, -1))  # (bs, head, seq_len, seq_len)
    att_logits /= d_k**0.5

    if is_casual:
        mask = th.triu(
            th.full(
                (att_logits.size(-1), att_logits.size(-1)),
                -1 * th.inf,
                device=att_logits.device,
            ),
            diagonal=1,
        )
        att_logits = att_logits + mask
    att_scores = F.softmax(att_logits, dim=-1)  # (bs, head, seq_len, seq_len)

    # (bs, head, seq_len, seq_len) * (bs, head, seq_len, dk) = (bs, head, seq_len, dk)
    return th.matmul(att_scores, v), att_scores
