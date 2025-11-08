#!/usr/bin/env python3
"""
PyTorch GPU Test Script
Tests GPU availability and performance with PyTorch
"""

import torch
import time

def test_pytorch_gpu():
    """Test PyTorch GPU setup"""
    
    print("=" * 50)
    print("🔥 PYTORCH GPU TEST")
    print("=" * 50)
    
    # Basic PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Set device
        device = torch.device('cuda:0')
        print(f"\n✅ Using device: {device}")
        
        # Performance test
        print("\n" + "=" * 30)
        print("🚀 GPU PERFORMANCE TEST")
        print("=" * 30)
        
        # Test matrix multiplication on GPU
        size = 2000
        print(f"Testing {size}x{size} matrix multiplication...")
        
        # GPU test
        a_gpu = torch.randn(size, size, device=device)
        b_gpu = torch.randn(size, size, device=device)
        
        # Warm up
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        # Actual test
        start_time = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"✅ GPU Time: {gpu_time*1000:.2f}ms")
        
        # CPU comparison
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start_time = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        
        print(f"🐌 CPU Time: {cpu_time*1000:.2f}ms")
        print(f"🚀 GPU Speedup: {cpu_time/gpu_time:.1f}x faster!")
        
        # Memory info
        print(f"\n📊 GPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
        return True
        
    else:
        print("❌ CUDA not available!")
        print("Possible solutions:")
        print("1. Install NVIDIA GPU drivers")
        print("2. Install CUDA Toolkit")
        print("3. Check if GPU is CUDA compatible")
        return False

if __name__ == "__main__":
    gpu_available = test_pytorch_gpu()
    
    if gpu_available:
        print(f"\n🎯 SUCCESS! GPU is ready for training!")
        print("Your MENINGIOMA detection will train FAST on GPU!")
    else:
        print(f"\n⚠️ GPU not available. Will use CPU.")