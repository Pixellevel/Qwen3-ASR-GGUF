import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from transformers.modeling_outputs import BaseModelOutput

# 导入必要的配置类
from configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from modeling_qwen3_asr import (
    Qwen3ASRAudioEncoder, 
    SinusoidsPositionEmbedding, 
    Qwen3ASRAudioEncoderLayer
)

class StatefulAudioEncoderWrapper(nn.Module):
    """
    Qwen3-ASR Audio Encoder 的 ONNX 导出包装类。
    支持流式推理中的卷积状态（Conv State）管理。
    """
    def __init__(self, encoder: Qwen3ASRAudioEncoder):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        
    def forward(
        self, 
        input_features: torch.Tensor,  # [batch, seq_len, 128]
        conv_state: torch.Tensor,      # [batch, overlap_len, 128]
        seq_offset: torch.Tensor,      # [1] or scalar
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 拼接状态
        full_features = torch.cat([conv_state, input_features], dim=1) 
        next_conv_state = full_features[:, -8:, :] # 取末尾作为状态
        
        # 2. 卷积下采样
        x = full_features.transpose(1, 2).unsqueeze(1)
        x = F.gelu(self.encoder.conv2d1(x))
        x = F.gelu(self.encoder.conv2d2(x))
        x = F.gelu(self.encoder.conv2d3(x))
        
        # 3. 展平并映射
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        hidden_states = self.encoder.conv_out(x)
        
        # 4. 位置编码 (关键修正：使用 seq_offset)
        # 此时 t 是包含 overlap token 的总长度。
        # Index 0 是 overlap token (对应 seq_offset - 1)
        # Index 1 是第一个 valid token (对应 seq_offset)
        # 我们构建 indices = [seq_offset-1, seq_offset, ..., seq_offset + t - 2]
        
        # 注意：seq_offset 是 Tensor，需要确保计算在正确的设备上
        start_pos = seq_offset - 1
        pos_indices = torch.arange(t, device=hidden_states.device) + start_pos
        # Clamp negative indices to 0 (针对 seq_offset=0 的情况，此时 overlap token 是 pad)
        pos_indices = torch.clamp(pos_indices, min=0)
        
        # 从 buffer 中 gather
        # positional_embedding shape: [max_len, dim]
        # F.embedding 需要 indices 是 LongTensor
        pos_emb = F.embedding(
            pos_indices, 
            self.encoder.positional_embedding.positional_embedding
        )
        
        hidden_states = hidden_states + pos_emb.unsqueeze(0).to(hidden_states.dtype)
        
        # 5. Transformer 处理 (关键：需要打平 Batch 维度以匹配官方 Attention 实现)
        # 官方实现中，hidden_states 是 [Total_Tokens, Hidden_Dim]
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)
        hidden_dim = hidden_states.size(2)
        
        hidden_states_flattened = hidden_states.view(-1, hidden_dim)
        
        # 生成 cu_seqlens
        # 因为我们是固定窗口，cu_seqlens 描述每个 Batch 的起始位置
        cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=hidden_states.device)
        
        for layer in self.encoder.layers:
            # layer 返回 (hidden_states,)
            hidden_states_flattened = layer(hidden_states_flattened, cu_seqlens)[0]
            
        # 恢复维度 [B, T, H]
        hidden_states = hidden_states_flattened.view(batch_size, seq_len, hidden_dim)
            
        hidden_states = self.encoder.ln_post(hidden_states)
        hidden_states = self.encoder.proj1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.encoder.proj2(hidden_states)
        
        # 6. Slicing (关键修正：去除首个 overlap token)
        hidden_states = hidden_states[:, 1:, :]
        
        return hidden_states, next_conv_state

class DiscreteAudioEncoder(nn.Module):
    """
    非流式的导出包装，用于验证。
    必须与官方 modeling_qwen3_asr.py 中的分块逻辑对齐，否则 Token 数量和边界特征会不一致。
    """
    def __init__(self, encoder: Qwen3ASRAudioEncoder):
        super().__init__()
        self.encoder = encoder
        self.n_window = encoder.config.n_window # 通常是 100 (对应 50 个 token)
        
    def forward(self, input_features: torch.Tensor):
        # input_features: [B, T, D]
        b, t, d = input_features.size()
        chunk_size = self.n_window * 2 # 100
        
        # 1. 分块 (对齐官方 forward L693)
        # 注意：ONNX 导出时，若 t 是动态的，split 可能会有问题。
        # 我们采用更稳妥的方式：Padding 到 chunk_size 的倍数，然后 Reshape
        pad_len = (chunk_size - (t % chunk_size)) % chunk_size
        if pad_len > 0:
            input_features = F.pad(input_features, (0, 0, 0, pad_len))
        
        t_padded = t + pad_len
        num_chunks = t_padded // chunk_size
        
        # [B, T_p, D] -> [B * N, chunk_size, D]
        # 模拟官方的 padded_feature [num_chunks, D, chunk_size]
        chunks = input_features.view(b * num_chunks, chunk_size, d)
        x = chunks.transpose(1, 2).unsqueeze(1) # [B*N, 1, D, chunk_size]
        
        # 2. 卷积下采样
        x = F.gelu(self.encoder.conv2d1(x))
        x = F.gelu(self.encoder.conv2d2(x))
        x = F.gelu(self.encoder.conv2d3(x))
        
        # 3. 展平与映射
        bn, c, f, tn = x.size() # tn 通常是 13 (针对 chunk_size=100)
        x = x.permute(0, 3, 1, 2).contiguous().view(bn, tn, c * f)
        chunk_hidden = self.encoder.conv_out(x) # [B*N, tn, H]
        
        # 4. 位置编码 (Chunk-wise)
        # 关键对齐：官方模型中，每个 100 帧块的卷积输出都加上从 0 开始的位置编码
        pos_emb = self.encoder.positional_embedding.positional_embedding[:tn, :]
        chunk_hidden = chunk_hidden + pos_emb.unsqueeze(0).to(chunk_hidden.dtype)
        
        # 5. 合并并还原形状
        hidden_states = chunk_hidden.view(b, num_chunks * tn, -1)
        
        # 6. 处理 Mask / 裁剪 (根据原始长度 t 计算真实的 token 数)
        # 模拟 _get_feat_extract_output_lengths
        input_lengths_leave = t % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1 if input_lengths_leave > 0 else 0
        rem_tokens = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 if feat_lengths > 0 else 0
        total_tokens = (t // 100) * 13 + rem_tokens
        
        # 切片，去除因为 Padding 产生的多余 token
        hidden_states = hidden_states[:, :total_tokens, :]
        
        # 7. Transformer 处理 (注意：官方使用 n_window_infer 进行 Attention 分块)
        # 默认 n_window_infer=400, 对应 52 个 token (4 * 13)
        window_tokens = 52 # 13 * (400 // 100)
        
        bh, th, dh = hidden_states.size()
        hidden_states_flattened = hidden_states.view(-1, dh)
        
        # 生成对齐官方的 cu_seqlens
        cu_chunk_lens = []
        for i in range(th // window_tokens):
            cu_chunk_lens.append(window_tokens)
        if th % window_tokens != 0:
            cu_chunk_lens.append(th % window_tokens)
            
        cu_seqlens = torch.tensor([0] + cu_chunk_lens, dtype=torch.int32, device=hidden_states.device).cumsum(0, dtype=torch.int32)
        
        for layer in self.encoder.layers:
            # 注意：官方层需要 flattened 输入和 cu_seqlens
            hidden_states_flattened = layer(hidden_states_flattened, cu_seqlens)[0]
            
        hidden_states = hidden_states_flattened.view(bh, th, dh)
        
        # 8. 后处理
        hidden_states = self.encoder.ln_post(hidden_states)
        hidden_states = self.encoder.proj1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.encoder.proj2(hidden_states)
        
        return hidden_states
