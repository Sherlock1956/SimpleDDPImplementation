import torch
import numpy as np
import timeit
def self_attention(Q, K, V):
    d = Q.shape[-1]
    attention = (Q @ K.transpose(-1, -2)) / d ** 0.5
    attention = torch.softmax(attention, dim=-1)
    return attention @ V
class Flash_attention_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        N_q, d = Q.shape[-2:]
        N_k = K.shape[-2]
        B_q = 64
        B_k = 192
        O = torch.zeros_like(Q)
        l = torch.zeros(Q.shape[:-1])
        m = torch.zeros(Q.shape[:-1]) - torch.inf
        L = torch.zeros(Q.shape[:-1])
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
if __name__ == "__main__":
    seq_len_list = [256, 1024, 4096, 8192, 16384]
    for _ in range(5):
        self_attention(torch.rand(1, 4096, 64),torch.rand(1, 4096, 64),torch.rand(1, 4096, 64))
    Q = torch.rand(8, 16384, 64)
    K = torch.rand(8, 16384, 64)
    V = torch.rand(8, 16384, 64)
    attention_pt = self_attention(Q, K, V)
    attention_flash = apply_flash_atn_pt(Q, K, V)
    assert torch.allclose(attention_flash, attention_pt, rtol=1e-5)
    for seq_len in seq_len_list:
        Q = torch.rand(8, seq_len, 64)
        K = torch.rand(8, seq_len, 64)
        V = torch.rand(8, seq_len, 64)
        pytorch_time = timeit.timeit(lambda: self_attention(Q, K, V), number=1)
        flash_time = timeit.timeit(lambda: apply_flash_atn_pt(Q, K, V), number=1)
        print(f"Seq_len: {seq_len}, PyTorch time: {pytorch_time:.4f}s, Flash time: {flash_time:.4f}s")