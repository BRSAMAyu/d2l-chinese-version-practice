import torch

# 檢查 PyTorch 版本
print(f"PyTorch Version: {torch.__version__}")

# 檢查 CUDA 是否可用
is_available = torch.cuda.is_available()
print(f"CUDA Available: {is_available}")

if is_available:
    # 顯示 GPU 數量
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    # 顯示目前使用的 GPU 索引
    print(f"Current GPU Index: {torch.cuda.current_device()}")
    # 顯示目前 GPU 的名稱
    print(f"Current GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")