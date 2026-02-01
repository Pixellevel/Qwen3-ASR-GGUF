import os
import numpy as np
import gguf

def get_token_embeddings_debug(model_path):
    print(f"--- 调试 GGUF Embedding 提取 ---")
    print(f"模型路径: {model_path}")
    
    reader = gguf.GGUFReader(model_path, mode='r')
    
    # 打印所有的 Fields (Metadata) 来观察结构
    print("\n[元数据识别]")
    # pip install 的 gguf 库中，GGUFReader 对象通常使用 .fields
    if hasattr(reader, 'fields'):
        print("✓ 发现 reader.fields")
        # 尝试查找 embedding 长度。注意：架构前缀可能不同
        for key, field in reader.fields.items():
            if "embedding_length" in key:
                n_embd = field.parts[-1][0] # GGUFValue 结构
                print(f"找到维度信息: {key} = {n_embd}")
    elif hasattr(reader, 'metadata'):
        print("✓ 发现 reader.metadata")
    else:
        print("❌ 既没有 fields 也没有 metadata 属性")

    # 遍历 Tensors
    print("\n[张量识别]")
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            print(f"找到 Embedding 张量: {t.name}")
            print(f"  - 类型: {t.tensor_type}")
            print(f"  - 形状: {t.shape}")
            
            # 读取数据
            # 对于 f16 类型，gguf 库通常会自动转换为 numpy.ndarray
            data = t.data
            print(f"  - 数据实际类型: {type(data)}")
            if isinstance(data, np.ndarray):
                print(f"  - Numpy Dtype: {data.dtype}")
                #如果是 f16，转为 f32 方便后续计算
                if data.dtype == np.float16:
                    data = data.astype(np.float32)
                return data
            else:
                print("  - ⚠️ 数据非 Numpy 数组，需要手动处理 buffer")
                
    return None

if __name__ == "__main__":
    MODEL_PATH = "model/qwen3_asr_llm.f16.gguf"
    if os.path.exists(MODEL_PATH):
        embd = get_token_embeddings_debug(MODEL_PATH)
        if embd is not None:
            print(f"\n✅ 提取成功! Shape: {embd.shape}")
        else:
            print("\n❌ 提取失败")
    else:
        print(f"❌ 找不到模型: {MODEL_PATH}")
