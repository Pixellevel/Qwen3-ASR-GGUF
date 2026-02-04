# Qwen3-ASR Encoder ONNX Dynamo 导出与 DirectML 优化经验

在使用 `torch.onnx.export` 的 `dynamo=True` (TorchDynamo) 模式导出 Qwen3-ASR 的音频编码器（Frontend 和 Backend）时，为了适配 **Windows DirectML** 运行时并解决常见的 `Reshape` 报错（如 `80070057`），总结了以下核心优化经验。

## 1. 消除 Python 控制流 (Graph Break 优化)
TorchDynamo 模式下，Python 的 `if/else`、`for` 循环（基于标量）会导致图分裂（Graph Break），生成的 ONNX 会包含碎裂的逻辑。

*   **错误做法**：在 `forward` 中使用 `if pad_len > 0: x = F.pad(...)`。
*   **优化方案**：使用纯 Tensor 运算。`F.pad` 的填充值如果是 0，其开销极小且能保持静态图逻辑的一致性。
*   **SymInt 陷阱**：动态长度在 Dynamo 下是符号化整数 (`SymInt`)。它不支持 `.clamp()` 等张量方法。应利用整数除法的特性（如 `-1 // 2 + 1 = 0`）来处理边界，而非使用 `clamp` 或 `if`。

## 2. “形状继承”技巧 (Shape Inheritance)
DirectML 非常自律，它不喜欢看到通过 `Shape -> Gather -> 算术运算 -> Concat -> Reshape` 产生的动态形状。

*   **错误做法**：使用 `torch.arange(t)` 生成位置编码。这会强制导出器去计算 `t` 的值。
*   **优化方案**：利用输入张量自带的形状信息。
    ```python
    # 推荐：DirectML 非常喜欢的写法，不产生显式的 Shape 计算节点
    positions = torch.ones_like(x[:, :, 0], dtype=torch.float32).cumsum(1) - 1
    ```

## 3. Reshape 的降维打击：`unflatten` 与 `flatten`
DirectML 报错 `node_view_2` 或 `Reshape` 失败，通常是因为 `view` 或 `reshape` 里的参数太复杂，导致生成的 ONNX `Reshape` 节点的第二个输入是一个“碎裂的 Concat”。

*   **核心原则**：**不要猜维度，要声明变换。**
*   **优化技巧**：
    *   **Head 切分**：使用 `unflatten(-1, (heads, head_dim))`，绝对不要去 `view` 中间的时间轴维度（如 `view(b, t, h, d)`）。
    *   **维度合并**：使用 `flatten(2)` 替代 `view(b, t, -1)`。
    *   **维重组（下采样）**：
        ```python
        # 推荐：先用 unflatten 切开，再用 flatten 合并非相邻维度
        x = x.unflatten(1, (num_chunks, chunk_size)).transpose(1, 2).flatten(2)
        ```

## 4. 导出参数对齐 (`dynamo=True`)
*   **参数切换**：在 `dynamo=True` 时，不要使用 `dynamic_axes`，而应改用 `dynamic_shapes`。
    ```python
    dynamic_shapes = {
        "mel": {
            0: torch.export.Dim("batch", min=1, max=16),
            1: torch.export.Dim("n_frames", min=1, max=4096)
        }
    }
    ```
*   **Opset 选择**：推荐使用 `opset_version=18`，它对复杂的形状计算有更好的原生算子支持。

## 5. 推理时的数据类型匹配 (TensorRT/DirectML)
*   **FP16 陷阱**：如果模型导出为 FP16 模式，输入数据必须显式调用 `.half()`。
*   **报错信息**：`Unexpected input data type. Actual: (tensor(float)) , expected: (tensor(float16))`。
*   **解决**：在 `transcriber` 的 `run` 之前，确保 `dummy_mel` 等输入张量已转为 `np.float16`。

## 6. 权重加载与自定义建模
在使用自定义的 `modeling_qwen3_asr.py` 代替原始库代码时，确保：
1.  将自定义目录添加到 `sys.path`。
2.  在导出脚本中显式从自定义文件 `import Qwen3ASRForConditionalGeneration`，否则 `from_pretrained` 可能会去加载缓存中的原始代码，导致你的优化逻辑失效。

## 7. ONNX 量化冲突定位 (896 vs 3584 报错)
在对 Transformer 后端进行 INT8 量化时，常遇到 `Inferred shape (896) vs existing shape (3584)` 错误。
*   **根源分析**：这是因为量化工具在处理复杂的**变长序列索引**（如 `cu_seqlens`）时，符号推断引擎“逻辑短路”，无法证明张量在经过 MLP 扩维后依然能与残差主干对齐，从而产生虚假约束冲突。
*   **传统解法（有风险）**：使用 `onnxsim` 简化模型。它能修复元数据，但往往会将模型“固定化”，导致在推理变长音频时 DirectML 报错 `8007023E` (Reshape 尺寸不匹配)。

## 8. 终极方案：3D 维度重构 (3D Dimension Reconstruction)
为了既能通过量化校验，又能完美适配 DirectML 的动态性，应采用 **“Padding -> 3D 变换 -> Transformer -> 还原”** 的策略：

1.  **Padding 对齐**：将输入序列 $T$ 填充到固定窗口大小（如 104）的倍数。
2.  **升维处理**：利用 `unflatten` 把维度从 `[B, T, H]` 变成 `[B * Num_Chunks, Window_Size, H]` 的 **3D 结构**。
3.  **静态窗口推理**：Transformer 每一层看到的输入长度固定（104），不再有任何符号计算，逻辑极其清晰。
4.  **还原并切除**：推理结束后 `flatten` 回 2D，并切除末尾的 Padding 部分。

*   **优势**：
    *   **量化友好**：每一层维度在图上显式固定，推断引擎秒过。
    *   **DML 友好**：算子图极其精简，无 `Range/Mod`，性能极佳 (RTF 可达 0.02)。
    *   **精度稳定**：无需使用 `onnxsim` 即可在 CPU/GPU 上全通。

## 9. onnxsim 与 DirectML 的兼容性陷阱
*   **注意**：`onnxsim` 在简化模型时，可能会通过 `constant folding` 将某些动态 `Reshape` 节点坍缩为固定数字。
*   **后果**：如果转录时长与导出时的 dummy 长度不一致，DirectML 会因为 Reshape 前后 Size 不等而崩溃。
*   **对策**：对于 DML 场景，优先通过**代码重构**（如 3D 重构）来简化逻辑，而非依赖外部工具。如果必须用 `onnxsim`，请确保不要使用 `overwrite_input_shapes` 固定输入。

---
*整理时间：2026-02-04 (Update: 针对 3D 重构方案)*
*整理者：Antigravity*
