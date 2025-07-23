import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os

def setup(rank, world_size):
    """初始化分布式进程组"""
    print(f"[Rank {rank}] 初始化分布式进程组")
    # 修正1: 使用环境变量或更稳定的初始化方式
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.set_default_device(f"cuda:{rank}")
    
def get_ipc_cuda_handle(tensor: torch.Tensor) -> tuple:
    """
    将GPU张量通过CUDA IPC方式共享，并返回句柄
    修正版本：增加错误检查和更好的兼容性
    """
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on CUDA device.")
    
    # 修正2: 确保tensor是连续的，避免stride问题
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    storage = tensor.untyped_storage()
    
    try:
        # 修正3: 添加错误处理
        handle = storage._share_cuda_()
        print(f"[DEBUG] storage._share_cuda_() returned handle with {len(handle)} elements")
    except Exception as e:
        print(f"[ERROR] Failed to share CUDA storage: {e}")
        raise
    
    return (
        handle,                        # storage handle: tuple of 8
        tensor.storage_offset(),
        tuple(tensor.size()),
        tuple(tensor.stride()),
        str(tensor.dtype).split(".")[-1],
        tensor.device.index
    )

def load_from_ipc_cuda_handle(handle: tuple) -> torch.Tensor:
    """
    通过CUDA IPC句柄在当前进程重建GPU张量
    修正版本：增加错误检查和更好的兼容性
    """
    try:
        storage_handle, storage_offset, size, stride, dtype_str, device_index = handle
        print(f"[DEBUG] Received storage_handle with {len(storage_handle)} elements")
        
        # 修正4: 更严格的数据类型检查
        if not hasattr(torch, dtype_str):
            raise ValueError(f"Invalid dtype string: {dtype_str}")
        
        dtype = getattr(torch, dtype_str)
        
        # 修正5: 添加设备检查
        current_device = torch.cuda.current_device()
        if device_index != current_device:
            print(f"[WARNING] Target device {device_index} != current device {current_device}")
        
        # 修正6: 使用更安全的存储重建方式
        try:
            storage = torch.UntypedStorage._new_shared_cuda(storage_handle, device_index)
        except Exception as e:
            print(f"[ERROR] Failed to create shared CUDA storage: {e}")
            raise
        
        # 修正7: 创建tensor时使用正确的设备
        tensor = torch.empty((0,), dtype=dtype, device=f"cuda:{device_index}")
        tensor = tensor.set_(storage, storage_offset, size, stride)
        
        return tensor
        
    except Exception as e:
        print(f"[ERROR] Failed to load from IPC handle: {e}")
        raise

def producer(rank, world_size, q, event):
    """生产者进程 - 修正版本"""
    try:
        setup(rank, world_size)
        print(f"[Producer {rank}] 启动成功")
        
        # 修正8: 使用更合理的张量大小，避免内存问题
        k_cache = torch.randn(8, 4, 256, 32, device=f"cuda:{rank}")  # 减小尺寸
        print(f"[Producer {rank}] 创建张量: shape={k_cache.shape}, device={k_cache.device}")
        
        # 修正9: 确保张量在正确的设备上
        assert k_cache.device.index == rank, f"Tensor on wrong device: {k_cache.device.index} != {rank}"
        
        # 获取共享内存句柄
        handle = get_ipc_cuda_handle(k_cache)
        print(f"[Producer {rank}] 获取IPC句柄成功")
        
        q.put(handle)
        event.set()
        
        # 修正10: 添加同步确保数据完整性
        torch.cuda.synchronize()
        
        dist.barrier()  # 同步等待消费者完成
        print(f"[Producer {rank}] 完成")
        
    except Exception as e:
        print(f"[Producer {rank}] 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def consumer(rank, world_size, q, event):
    """消费者进程 - 修正版本"""
    try:
        setup(rank, world_size)
        print(f"[Consumer {rank}] 启动成功")
        
        event.wait()
        handle = q.get()
        
        # 修正11: 添加同步确保句柄有效
        torch.cuda.synchronize()
        
        tensor = load_from_ipc_cuda_handle(handle)
        print(f"[Consumer {rank}] 成功加载张量: device={tensor.device}, shape={tensor.shape}")
        
        # 修正12: 验证数据完整性
        if tensor.numel() > 0:
            print(f"[Consumer {rank}] 张量统计: mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")
        
        dist.barrier()
        print(f"[Consumer {rank}] 完成")
        
    except Exception as e:
        print(f"[Consumer {rank}] 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    """主函数 - 修正版本"""
    # 修正13: 添加CUDA可用性检查
    if not torch.cuda.is_available():
        print("CUDA不可用，无法运行此示例")
        return
    
    if torch.cuda.device_count() < 2:
        print("需要至少2个CUDA设备来运行此示例")
        return
    
    world_size = 2  # 两个GPU进程
    
    # 修正14: 使用spawn启动方法，避免CUDA fork问题
    mp.set_start_method('spawn', force=True)
    
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    event = ctx.Event()
    
    try:
        process1 = ctx.Process(target=producer, args=(0, world_size, q, event))
        process2 = ctx.Process(target=consumer, args=(1, world_size, q, event))
        
        process1.start()
        process2.start()
        
        process1.join(timeout=30)  # 修正15: 添加超时
        process2.join(timeout=30)
        
        if process1.is_alive():
            print("Producer进程超时，强制终止")
            process1.terminate()
            process1.join()
        
        if process2.is_alive():
            print("Consumer进程超时，强制终止")
            process2.terminate()
            process2.join()
            
        print("所有进程完成")
        
    except Exception as e:
        print(f"主进程错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()