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
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, dO_ptr, dQ_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
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
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (query_tile_index*Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_qb,
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
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES, ),
        strides = (stride_lq, ),
        offsets = (query_tile_index*Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,),
    )    
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape = (N_QUERIES, ),
        strides = (stride_dq, ),
        offsets = (query_tile_index*Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,),
    )
    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_optino="zero")
    dS = tl.zeros((Q_TILE_SIZE, D),dtype=tl.float32)
    dP = tl.zeros((Q_TILE_SIZE, D),dtype=tl.float32)
    S = tl.zeros((Q_TILE_SIZE, D),dtype=tl.float32)
    P = tl.zeros((Q_TILE_SIZE, D),dtype=tl.float32)
    dQ = tl.zeros((Q_TILE_SIZE, D),dtype=tl.float32)
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        l = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        S = tl.dot(Q, tl.trans(K)) * scale
        P = tl.exp(S - l[:,None])
        dP = tl.dot(dO, tl.trans(V))
        dS = P * (dP - D_i[:,None])
        dQ += tl.dot(dS, K) * scale
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    tl.store(dQ_block_ptr, boundary_check=(0, 1))
@triton.jit
def flash_bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, dO_ptr, dK_ptr, dV_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (0, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (0, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets = (key_tile_index*Q_TILE_SIZE, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets = (key_tile_index*Q_TILE_SIZE, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets = (key_tile_index*K_TILE_SIZE, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets = (key_tile_index*K_TILE_SIZE, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES, ),
        strides = (stride_lq, ),
        offsets = (0,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,),
    )    
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape = (N_QUERIES, ),
        strides = (stride_dq, ),
        offsets = (0,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,),
    )
    K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
    V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)

    dS = tl.zeros((Q_TILE_SIZE, D),dtype=tl.float32)
    dP = tl.zeros((Q_TILE_SIZE, D),dtype=tl.float32)
    S = tl.zeros((Q_TILE_SIZE, D),dtype=tl.float32)
    P = tl.zeros((Q_TILE_SIZE, D),dtype=tl.float32)
    dK = tl.zeros((K_TILE_SIZE, D),dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE, D),dtype=tl.float32)
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D),
        dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D),
        D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_optino="zero")# (K_TILE_SIZE,),
        l = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")

        S = tl.dot(Q, tl.trans(K)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        P = tl.exp(S - l[:,None])
        dV += tl.dot(tl.trans(P),dO) # (K_TILE_SIZE, D)
        dP = tl.dot(dO, tl.trans(V)) # (Q_TILE_SIZE, K_TILE_SIZE)
        dS = P * (dP - D_i[:,None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        dK += tl.dot(tl.trans(dS),Q) * scale
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
    tl.store(dK_block_ptr, boundary_check=(0, 1))
    tl.store(dV_block_ptr, boundary_check=(0, 1))
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
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
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
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32) # 需要初始化，注o数据类型
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,),  -float("inf"), dtype=tl.float32) # tl.full的使用，-float("inf")
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Q_i = tl.cast(Q_i, tl.float32)
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        j_start = j * K_TILE_SIZE  
        # 创建 key index 向量并判断哪些是有效（< N_KEYS）
        k_idx = tl.arange(0, K_TILE_SIZE) + j_start  # shape (K_TILE_SIZE,)
        valid_k = k_idx < N_KEYS                      # boolean mask shape (K_TILE_SIZE,)
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        K_j = tl.cast(K_j, tl.float32) # 数据类型
        V_j = tl.cast(V_j, tl.float32)
        S_i = tl.dot(Q_i, tl.trans(K_j)) * scale
        # mask operation 
        if is_causal:
            # 计算块中元素在S中的的行数
            q_idx = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE  # shape (Q_TILE_SIZE,)
            # 计算块中元素在S中的的列数 
            k_idx_tile = tl.arange(0, K_TILE_SIZE) + j_start  # shape (K_TILE_SIZE,)
            # 行数大于列数就被mask，合理利用广播机制
            causal_mask = q_idx[:, None] >= k_idx_tile[None, :]  # shape (Q_TILE_SIZE, K_TILE_SIZE)
            # Apply causal mask: add -1e6 to masked out elements
            S_i = tl.where(causal_mask, S_i, S_i - 1e6)
        S_i = tl.where(valid_k[None,:], S_i, -float("inf"))
        m_i_old = m_i
        m_i = tl.maximum(m_i_old, tl.max(S_i, axis=-1))
        P_i = tl.exp(S_i - m_i[:, None]) # triton中不能用...,None
        l_i = tl.exp(m_i_old - m_i) * l_i + tl.sum(P_i, axis=-1)
        O_i = tl.exp(m_i_old - m_i)[:, None] * O_i + tl.dot(P_i, V_j)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)
    tl.store(O_block_ptr, tl.cast(O_i, tl.float32), boundary_check=(0, 1))
    tl.store(L_block_ptr, tl.cast(L_i, tl.float32), boundary_check=(0,))
class Flash_attention_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
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
        tile_size = 32 if D > 64 else 64
        Q_TILE_SIZE = 32  # Reduced from 256 to 64， 不能太大了，shared memory空间有限
        K_TILE_SIZE = 32 # Reduced from 256 to 64
        grid = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)
        ctx.is_causal = is_causal
        ctx.scale = scale
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D,
            Q_TILE_SIZE = Q_TILE_SIZE,
            K_TILE_SIZE = K_TILE_SIZE,
            is_causal = is_causal
        )
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_O):
        L, Q, K, V, O = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal
        batch_size,n_queries = Q.shape[:-1]
        n_keys = K.shape[-2]
        # D_i should be the sum of element-wise multiplication along the last dimension
        D_i = torch.sum(O * grad_O, dim=-1)  # Shape: (batch_size, seq_len)
        S = (Q @ K.transpose(-1, -2)) * scale # 需要对S进行causal mask，后面的P_ij, dS_ij自动就mask了
        if is_causal:
            causal_mask = torch.arange(0,n_queries)[...,None] >= torch.arange(0,n_keys)[...,None,:]
            causal_mask = causal_mask.to(Q.device)
            S = torch.where(causal_mask, S, torch.full_like(S, -1e6))
        P_ij = torch.exp(S - L[...,None])
        dV = P_ij.transpose(-1, -2) @ grad_O
        dP = grad_O @ V.transpose(-1, -2)
        dS_ij = P_ij * (dP - D_i[...,None])
        dQ = (dS_ij @ K) * scale
        dK = (dS_ij.transpose(-1, -2) @ Q) * scale
        return dQ, dK, dV, None # 输出的梯度个数必须与输入参数数量一致，is_causal的梯度应该是None
class Flash_attention_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        ctx.is_causal = is_causal
        N_q, d = Q.shape[-2:]
        ctx.scale = 1 / d ** 0.5
        N_k = K.shape[-2]
        B_q = 32
        B_k = 32  # Reduced to match Triton implementation
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
    def backward(ctx, grad_O):
        L, Q, K, V, O = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal
        batch_size,n_queries = Q.shape[:-1]
        n_keys = K.shape[0-2]
        # D_i should be the sum of element-wise multiplication along the last dimension
        D_i = torch.sum(O * grad_O, dim=-1)  # Shape: (batch_size, seq_len)
        S = (Q @ K.transpose(-1, -2)) * scale
        P_ij = torch.exp(S - L[...,None])
        dV = P_ij.transpose(-1, -2) @ grad_O
        dP = grad_O @ V.transpose(-1, -2)
        dS_ij = P_ij * (dP - D_i[...,None])
        if is_causal:
            causal_mask = torch.arange(0,n_queries)[...,None] >= torch.arange(0,n_keys)[...,None,:]
            causal_mask = causal_mask.to(Q.device)
            dS_ij = torch.where(causal_mask, dS_ij, torch.zeros_like(dS_ij))
        dQ = (dS_ij @ K) * scale
        dK = (dS_ij.transpose(-1, -2) @ Q) * scale
        return dQ, dK, dV, None # 输出的梯度个数必须与输入参数数量一致，is_causal的梯度应该是None
def apply_flash_atn_pt(Q, K, V, is_causal):
    return Flash_attention_pytorch.apply(Q, K, V, is_causal)
def apply_flash_atn_triton(Q, K, V, is_causal):
    return Flash_attention_triton.apply(Q, K, V, is_causal)
if __name__ == "__main__":
    # compare forward pass time of flash attention and regular attention
    # device = 'cuda'
    # seq_len_list = [256, 1024, 4096, 8192, 16384, 16384*2, 16384*4]
    # Q = torch.rand(4, 1024, 64).to(device)
    # K = torch.rand(4, 1024, 64).to(device)
    # V = torch.rand(4, 1024, 64).to(device)
    # for _ in range(5):
    #     attention_pt = self_attention(Q, K, V)
    #     attention_flash = apply_flash_atn_triton(Q, K, V)
    # assert torch.allclose(attention_flash, attention_pt, rtol=1e-2, atol=1e-2)
    # for seq_len in seq_len_list:
    #     Q = torch.rand(8, seq_len, 64).to(device)
    #     K = torch.rand(8, seq_len, 64).to(device)
    #     V = torch.rand(8, seq_len, 64).to(device)
    #     pytorch_time = timeit.timeit(lambda: self_attention(Q, K, V), number=10)
    #     flash_time = timeit.timeit(lambda: apply_flash_atn_triton(Q, K, V), number=10)
    #     print(f"Seq_len: {seq_len}, PyTorch time: {pytorch_time:.4f}s, Flash time: {flash_time:.4f}s")

    # test backward pass
    device = 'cuda'
    Q = torch.rand(4, 1024, 64, requires_grad=True).to(device)# 需要写requires_grad
    K = torch.rand(4, 1024, 64, requires_grad=True).to(device)
    V = torch.rand(4, 1024, 64, requires_grad=True).to(device)
    attention = apply_flash_atn_pt(Q, K, V, True)
    loss = attention.sum()  # Use sum() instead of ** 2 for a scalar loss
    loss.backward()