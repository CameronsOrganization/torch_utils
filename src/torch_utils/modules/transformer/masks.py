from torch.nn.attention import flex_attention


def causal_mask(
    q_len: int,
    kv_len: int,
):
    def causal(batch_size, num_heads, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = flex_attention.create_block_mask(
        causal,
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device="cuda",
        _compile=True,
    )
    return block_mask
