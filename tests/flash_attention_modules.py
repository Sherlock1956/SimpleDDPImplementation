import torch
import numpy as np
import timeit
import triton
import triton.language as tl
def self_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (query_tile_index*Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets = (0, 0), # K, V的offsets都应该从0开始，随j遍历
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets = (0, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape = (N_QUERIES, D),
        strides = (stride_oq, stride_od),
        offsets = (query_tile_index*Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES, ),
        strides = (stride_lq, ),
        offsets = (query_tile_index*Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,),
    )
    O_i = tl.zeros((Q_TILE_SIZE, D)) # 需要初始化
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -tl.inf, dtype=tl.float32) # tl.full的使用
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Q_i = tl.cast(Q_i, tl.float32)
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        K_j = tl.cast(K_j, tl.float32) # 数据类型
        V_j = tl.cast(V_j, tl.float32)
        S_i = tl.dot(Q_i, tl.trans(K_j)) * scale
        # mask operation to be done

        ###########################
        m_i_old = m_i
        m_i = tl.maximum(m_i_old, tl.max(S_i, axis=-1))
        P_i = tl.exp(S_i - m_i[...,None])
        l_i = tl.exp(m_i_old - m_i) * l_i + tl.sum(P_i,axis=-1)
        O_i = tl.exp(m_i_old - m_i)[...,None] * O_i + tl.dot(P_i, V_j)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    O_i = O_i / l_i[...,None]
    L_i = m_i + tl.log(l_i)
    tl.store(O_block_ptr, tl.cast(O_i, tl.float32), boundary_check=(0, 1))
    tl.store(L_block_ptr, tl.cast(L_i, tl.float32), boundary_check=(0,))
class Flash_attention_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Tensors must be on CUDA"
        # ensure contiguous to make strides consistent
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        batch_size, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[-2]
        scale = 1 / (D ** 0.5)
        O = torch.empty((batch_size, N_QUERIES, D), device=Q.device, dtype=Q.dtype)
        L = torch.empty((batch_size, N_QUERIES), device=Q.device, dtype=Q.dtype)
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 192
        grid = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            scale,
            Q_TILE_SIZE = Q_TILE_SIZE,
            K_TILE_SIZE = K_TILE_SIZE
        )
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    def backward(ctx):
        raise NotImplementedError
class Flash_attention_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        N_q, d = Q.shape[-2:]
        N_k = K.shape[-2]
        B_q = 64
        B_k = 192
        O = torch.zeros_like(Q,device=Q.device)
        l = torch.zeros(Q.shape[:-1],device=Q.device)
        m = torch.zeros(Q.shape[:-1],device=Q.device) - torch.inf
        L = torch.zeros(Q.shape[:-1],device=Q.device)
        i_max = int(np.ceil(N_q / B_q))
        j_max = int(np.ceil(N_k / B_k))
        for i in range(i_max):
            Q_i = Q[...,i*B_q:(i+1)*B_q,:]
            O_i = O[...,i*B_q:(i+1)*B_q,:]
            l_i = l[...,i*B_q:(i+1)*B_q]
            m_i = m[...,i*B_q:(i+1)*B_q]
            L_i = L[...,i*B_q:(i+1)*B_q]
            for j in range(j_max):
                K_j = K[...,j*B_k:(j+1)*B_k,:]
                V_j = V[...,j*B_k:(j+1)*B_k,:]
                S_i = (Q_i @ K_j.transpose(-1,-2)) / d ** 0.5
                m_i_old = m_i.clone()
                m_i = torch.max(m_i_old,torch.max(S_i,dim=-1)[0])
                P_i = torch.exp(S_i - m_i[...,None])
                l_i = torch.exp(m_i_old - m_i) * l_i + torch.sum(P_i,dim=-1,keepdim=False)
                O_i = torch.exp(m_i_old - m_i)[...,None] * O_i + P_i @ V_j
                m[...,i*B_q:(i+1)*B_q] = m_i
                l[...,i*B_q:(i+1)*B_q] = l_i
            O_i = O_i / l_i[...,None]
            O[...,i*B_q:(i+1)*B_q,:] = O_i
            L_i = m_i + torch.log(l_i)
            L[...,i*B_q:(i+1)*B_q] = L_i
        ctx.save_for_backward(L, Q, K, V, O)
        return O
    @staticmethod
    def backward(ctx):
        raise NotImplementedError
def apply_flash_atn_pt(Q, K, V):
    return Flash_attention_pytorch.apply(Q, K, V)
def apply_flash_atn_triton(Q, K, V):
    return Flash_attention_triton.apply(Q, K, V)
if __name__ == "__main__":
    device = 'mps'
    seq_len_list = [256, 1024, 4096, 8192, 16384]
    Q = torch.rand(4, 16384, 64).to(device)
    K = torch.rand(4, 16384, 64).to(device)
    V = torch.rand(4, 16384, 64).to(device)
    for _ in range(5):
        attention_pt = self_attention(Q, K, V)
        attention_flash = apply_flash_atn_triton(Q, K, V)
    assert torch.allclose(attention_flash, attention_pt, rtol=1e-5)
    for seq_len in seq_len_list:
        Q = torch.rand(8, seq_len, 64).to(device)
        K = torch.rand(8, seq_len, 64).to(device)
        V = torch.rand(8, seq_len, 64).to(device)
        pytorch_time = timeit.timeit(lambda: self_attention(Q, K, V), number=1)
        flash_time = timeit.timeit(lambda: apply_flash_atn_triton(Q, K, V), number=1)
        print(f"Seq_len: {seq_len}, PyTorch time: {pytorch_time:.4f}s, Flash time: {flash_time:.4f}s")