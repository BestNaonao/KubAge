import torch
import time
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModel, AutoConfig
import os
import logging

# 配置日志，捕获transformers内部信息
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模型路径
qwen_path = "../models/Qwen/Qwen3-Embedding-0.6B"   # 替换为实际路径
bge_path = "D:/学习资料/毕业设计/KubAge/models/BAAI/bge-large-zh-v1___5"
qwen_path = bge_path

def verify_flash_attention():
    print("=" * 80)
    print("🔍 FLASH ATTENTION 验证流程")
    print("=" * 80)

    # 第一步：环境基础检查
    print("\n1️⃣ 环境基础检查")
    print(f"  • PyTorch 版本: {torch.__version__}")
    print(f"  • 是否使用C11ABI编译: {torch.compiled_with_cxx11_abi()}")
    print(f"  • CUDA 可用: {torch.cuda.is_available()}")
    print(f"  • CUDA 版本: {torch.version.cuda}")
    print(f"  • GPU 型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"  • GPU 内存: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"  • GPU 架构: {torch.cuda.get_device_capability() if torch.cuda.is_available() else 'N/A'}")
    print(torch.cuda.get_device_properties(0))

    # 检查 flash-attn 是否安装
    try:
        import flash_attn
        print(f"  • FlashAttention 版本: {flash_attn.__version__}")
    except ImportError as e:
        print(f"  ❌ FlashAttention 未安装: {str(e)}")
        return False

    # 第二步：检查模型配置
    print("\n2️⃣ 模型配置检查")
    try:
        config = AutoConfig.from_pretrained(qwen_path, trust_remote_code=True)
        print(f"  • 模型架构: {config.model_type}")

        # 检查是否支持 Flash Attention 2
        supports_flash2 = getattr(config, "_supports_flash_attn_2", False)
        print(f"  • 声明支持 Flash Attention 2: {'✅ 是' if supports_flash2 else '❌ 否'}")

        # 检查模型文件中是否包含 flash_attn 相关代码
        flash_files = [f for f in os.listdir(qwen_path) if "flash" in f.lower()]
        print(f"  • 模型目录中包含 Flash 相关文件: {'✅ ' + str(flash_files) if flash_files else '❌ 无'}")

    except Exception as e:
        print(f"  ❌ 配置检查失败: {str(e)}")

    return True

    # 第三步：尝试加载启用 Flash Attention 的模型
    print("\n3️⃣ 尝试加载启用 Flash Attention 的模型")
    try:
        # 创建启用 Flash Attention 的 embeddings
        embeddings_flash = HuggingFaceEmbeddings(
            model_name=qwen_path,
            model_kwargs={
                "device": "cuda",
                "trust_remote_code": True,
                "use_flash_attention_2": True  # 关键参数
            },
            encode_kwargs={"normalize_embeddings": True}
        )
        print("  ✅ 成功加载启用 Flash Attention 的模型")

        # 获取底层模型
        base_model = embeddings_flash.client._model
        print(f"  • 底层模型类型: {type(base_model)}")

        # 检查注意力层类型
        attention_layers = []
        for name, module in base_model.named_modules():
            if "attention" in name.lower():
                attention_layers.append((name, type(module).__name__))

        print(f"  • 检测到 {len(attention_layers)} 个注意力层")
        for name, layer_type in attention_layers[:3]:  # 只显示前3个
            print(f"    - {name}: {layer_type}")

        # 特别检查是否包含 FlashAttention
        has_flash = any("FlashSelfAttention" in str(type(module))
                        for _, module in base_model.named_modules())
        print(f"  • 包含 FlashAttention 层: {'✅ 是' if has_flash else '❌ 否'}")

    except Exception as e:
        print(f"  ❌ 加载失败: {str(e)}")
        if "does not support Flash Attention 2" in str(e):
            print("    ⚠️  模型架构不支持 Flash Attention 2")
        return False

    # 第四步：运行时验证（实际前向传播）
    print("\n4️⃣ 运行时验证 (实际推理)")
    try:
        # 创建测试文本
        test_texts = ["这是一个用于验证Flash Attention的测试文本。"] * 8  # 批量测试

        # 清理GPU缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 启用Flash Attention的推理
        start_time = time.time()
        flash_embeddings = embeddings_flash.embed_documents(test_texts)
        flash_time = time.time() - start_time
        flash_mem = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB

        print(f"  ✅ Flash Attention 推理成功!")
        print(f"    • 耗时: {flash_time:.4f} 秒")
        print(f"    • 峰值内存: {flash_mem:.2f} GB")
        print(f"    • 输出维度: {len(flash_embeddings[0])}")

    except Exception as e:
        print(f"  ❌ 推理失败: {str(e)}")
        return False

    # 第五步：对比验证（禁用Flash Attention）
    print("\n5️⃣ 对比验证 (禁用Flash Attention)")
    try:
        # 创建禁用Flash Attention的embeddings
        embeddings_normal = HuggingFaceEmbeddings(
            model_name=qwen_path,
            model_kwargs={
                "device": "cuda",
                "trust_remote_code": True,
                # 不启用 Flash Attention
            },
            encode_kwargs={"normalize_embeddings": True}
        )

        # 清理GPU缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 禁用Flash Attention的推理
        start_time = time.time()
        normal_embeddings = embeddings_normal.embed_documents(test_texts)
        normal_time = time.time() - start_time
        normal_mem = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB

        print(f"  ✅ 标准注意力推理成功!")
        print(f"    • 耗时: {normal_time:.4f} 秒")
        print(f"    • 峰值内存: {normal_mem:.2f} GB")

        # 性能对比
        speedup = normal_time / flash_time if flash_time > 0 else 0
        mem_saving = (normal_mem - flash_mem) / normal_mem * 100 if normal_mem > 0 else 0

        print(f"\n📊 性能对比结果:")
        print(f"    • 速度提升: {speedup:.2f}x ({normal_time:.4f}s → {flash_time:.4f}s)")
        print(f"    • 内存节省: {mem_saving:.1f}% ({normal_mem:.2f}GB → {flash_mem:.2f}GB)")

        # 验证输出一致性
        import numpy as np
        flash_arr = np.array(flash_embeddings)
        normal_arr = np.array(normal_embeddings)
        cos_sim = np.mean(np.sum(flash_arr * normal_arr, axis=1) /
                          (np.linalg.norm(flash_arr, axis=1) * np.linalg.norm(normal_arr, axis=1)))

        print(f"    • 输出一致性 (余弦相似度): {cos_sim:.6f}")
        if cos_sim > 0.999:
            print("    ✅ 输出高度一致，验证有效")
        else:
            print("    ⚠️  输出差异较大，可能验证不准确")

        # 判断是否真正使用了Flash Attention
        if speedup > 1.2 and mem_saving > 10 and cos_sim > 0.99:
            print("\n🎉 验证结论: 模型成功使用了 Flash Attention!")
            return True
        else:
            print("\n❌ 验证结论: 未检测到 Flash Attention 的实际效果")
            print("  可能原因:")
            print("  1. 嵌入模型只使用浅层，未触发完整注意力计算")
            print("  2. 输入序列太短，Flash Attention 优势不明显")
            print("  3. 模型架构不完全支持 Flash Attention 2")
            return False

    except Exception as e:
        print(f"  ❌ 对比验证失败: {str(e)}")
        return False


# 执行验证
if __name__ == "__main__":
    result = verify_flash_attention()
    print("\n" + "=" * 80)
    print(f"最终验证结果: {'✅ 成功' if result else '❌ 失败'}")
    print("=" * 80)