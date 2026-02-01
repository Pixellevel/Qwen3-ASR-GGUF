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

class EncoderConvFrontend(nn.Module):
    """
    Qwen3-ASR 音频编码器前端：分块卷积部分。
    将 Mel 频谱转换为中间隐藏状态。
    严格对齐官方的 100 帧物理分块逻辑，确保 Token 数量精确一致。
    """
    def __init__(self, encoder: Qwen3ASRAudioEncoder):
        super().__init__()
        self.encoder = encoder
        # 根据官方配置，n_window=50, 所以分块大小为 2 * n_window = 100
        self.chunk_size = 100 

    def forward(self, mel: torch.Tensor):
        # mel: [B, T, D]
        b, t, d = mel.size()
        chunk_size = self.chunk_size
        
        # 1. 物理分块与填充 (对齐官方行为)
        pad_len = (chunk_size - (t % chunk_size)) % chunk_size
        if pad_len > 0:
            mel = F.pad(mel, (0, 0, 0, pad_len))
        
        t_padded = t + pad_len
        num_chunks = t_padded // chunk_size
        
        # [B, T_p, D] -> [B * N, chunk_size, D]
        chunks = mel.view(b * num_chunks, chunk_size, d)
        x = chunks.transpose(1, 2).unsqueeze(1) # [B*N, 1, D, chunk_size]
        
        # 2. 卷积下采样
        x = F.gelu(self.encoder.conv2d1(x))
        x = F.gelu(self.encoder.conv2d2(x))
        x = F.gelu(self.encoder.conv2d3(x))
        
        # 3. 展平与映射
        bn, c, f, tn = x.size() # tn = 13
        x = x.permute(0, 3, 1, 2).contiguous().view(bn, tn, c * f)
        hidden_states = self.encoder.conv_out(x) # [B*N, 13, H]
        
        # 4. 合并并还原形状到 [B, N*13, H]
        hidden_states = hidden_states.view(b, num_chunks * tn, -1)
        
        # 5. 长度对齐 (移除末尾因为 Padding 产生的多余 Token)
        input_lengths_leave = t % chunk_size
        feat_lengths = (input_lengths_leave - 1) // 2 + 1 if input_lengths_leave > 0 else 0
        rem_tokens = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 if feat_lengths > 0 else 0
        total_tokens = (t // chunk_size) * tn + rem_tokens
        
        hidden_states = hidden_states[:, :total_tokens, :]
        return hidden_states

class EncoderTransformerBackend(nn.Module):
    """
    Qwen3-ASR 音频编码器后端：Transformer 部分。
    实现局部位置编码 (0-12) 和 104-token 窗口化注意力。
    """
    def __init__(self, encoder: Qwen3ASRAudioEncoder):
        super().__init__()
        self.encoder = encoder
        
    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: [B, T, H]
        b, t, h = hidden_states.size()
        device = hidden_states.device
        
        # 1. 添加局部位置编码 (对齐官方逻辑：每个分块内部重置位置索引)
        # 官方代码中是在 [Num_Chunks, 13, H] 形状下加法的
        tn = 13
        # 构造一个足够长的 position_embedding 映射
        # 我们按照 tn=13 进行周期性填充，或者简单reshape
        # 考虑到动态 T 可能不是 13 的倍数（虽然我们在 FE 做了处理），更稳妥的做法是：
        pos_indices = torch.arange(t, device=device) % tn
        pos_emb = F.embedding(
            pos_indices, 
            self.encoder.positional_embedding.positional_embedding
        )
        hidden_states = hidden_states + pos_emb.unsqueeze(0).to(hidden_states.dtype)
        
        # 2. Transformer 处理 (窗口化 Attention)
        # n_window_infer = 800 frames = 8 chunks * 13 tokens = 104 tokens
        window_size = 104 
        
        hidden_states_flattened = hidden_states.view(-1, h)
        
        # 生成符号化 cu_seqlens 以支持 ONNX 导出
        num_windows = (t + window_size - 1) // window_size
        steps = torch.arange(0, num_windows, device=device) * window_size
        cu_seqlens_single = torch.cat([steps, torch.tensor([t], device=device, dtype=steps.dtype)])
        
        if b > 1:
            offsets = (torch.arange(0, b, device=device) * t).view(-1, 1)
            cu_seqlens = (cu_seqlens_single.view(1, -1) + offsets).view(-1).to(torch.int32)
        else:
            cu_seqlens = cu_seqlens_single.to(torch.int32)
            
        for layer in self.encoder.layers:
            hidden_states_flattened = layer(hidden_states_flattened, cu_seqlens)[0]
            
        hidden_states = hidden_states_flattened.view(b, t, h)
        
        # 3. 后投影
        hidden_states = self.encoder.ln_post(hidden_states)
        hidden_states = self.encoder.proj1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.encoder.proj2(hidden_states)
        
        return hidden_states

class StatefulAudioEncoderWrapper(nn.Module):
    """
    为了向后兼容 transcribe.py 的 API，保留这个类。
    由于 Qwen3-ASR 卷积是固定分块且独立处理，流式状态仅用于缓存不满足 100 帧的残余物理帧。
    但在目前非流式转录中，它直接调用 Frontend + Backend。
    """
    def __init__(self, encoder: Qwen3ASRAudioEncoder):
        super().__init__()
        self.frontend = EncoderConvFrontend(encoder)
        self.backend = EncoderTransformerBackend(encoder)
        
    def forward(self, mel: torch.Tensor, conv_state: torch.Tensor = None, seq_offset: torch.Tensor = None):
        # 注意：这里的 conv_state 和 seq_offset 目前在 ASR 逻辑中不再是卷积必需
        # 我们暂时忽略它们以确保数学对齐，未来若做低延迟流式需重新设计缓存 buffer
        hidden_states_fe = self.frontend(mel)
        output = self.backend(hidden_states_fe)
        # 返回格式尽量兼容：(output, next_conv_state)
        # 由于这里不再使用 conv_state，我们返回一个假的 next_conv_state
        return output, torch.zeros((1, 8, 128), device=mel.device)

class DiscreteAudioEncoder(nn.Module):
    """
    全量合体版本。已在 01-合体encoder.py 中验证。
    """
    def __init__(self, encoder: Qwen3ASRAudioEncoder):
        super().__init__()
        self.fe = EncoderConvFrontend(encoder)
        self.be = EncoderTransformerBackend(encoder)
        
    def forward(self, mel: torch.Tensor):
        x = self.fe(mel)
        return self.be(x)
