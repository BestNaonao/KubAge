import torch
import flash_attn


# 检查GPU可用性
if not torch.cuda.is_available():
    print("CUDA不可用，请检查GPU配置")
    exit()

print(f"PyTorch版本: {torch.__version__}")
print(f"是否使用C11ABI编译: {torch.compiled_with_cxx11_abi()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU型号: {torch.cuda.get_device_name(0)}")
print(f"当前GPU内存: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
print(f"FlashAttention版本: {flash_attn.__version__}")

# 创建测试数据 - 必须使用fp16或bf16
batch_size, seqlen, num_heads, head_dim = 2, 128, 8, 64

# 选项1: 使用 float16 (推荐)
q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.float16).cuda()
k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.float16).cuda()
v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.float16).cuda()

# 选项2: 使用 bfloat16 (如果硬件支持)
# q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.bfloat16).cuda()
# k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.bfloat16).cuda()
# v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.bfloat16).cuda()

try:
    # 测试flash attention
    out = flash_attn.flash_attn_func(q, k, v)
    print("✅ FlashAttention测试成功!")
    print(f"输出形状: {out.shape}")
    print(f"输出数据类型: {out.dtype}")

    # 验证输出是否合理
    if torch.isnan(out).any():
        print("⚠️ 警告: 输出包含NaN值")
    else:
        print("✅ 输出数据正常，无NaN值")

except RuntimeError as e:
    print(f"❌ FlashAttention测试失败: {str(e)}")
    print("常见问题排查:")
    print("1. 确保输入数据是fp16或bf16类型")
    print("2. 检查CUDA版本是否兼容")
    print("3. 确认GPU架构是否支持(需要Ampere架构或更新)")

except Exception as e:
    print(f"❌ 发生未知错误: {str(e)}")

# 清理GPU内存
torch.cuda.empty_cache()
print("✅ GPU内存已清理")