#!/usr/bin/env python3
"""
GPU Setup and Verification Script
Checks GPU availability and configures TensorFlow for optimal performance
"""

import tensorflow as tf
import os

def check_gpu_setup():
    """Check and configure GPU for TensorFlow"""
    
    print("=" * 50)
    print("🔧 GPU SETUP AND VERIFICATION")
    print("=" * 50)
    
    # Basic TensorFlow info
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPU Available: {len(gpus)} GPU(s) detected")
    
    if gpus:
        print("✅ GPU DETECTED!")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
        
        # Configure GPU memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
            
            # Enable mixed precision for faster training
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print(f"✅ Mixed precision enabled: {policy.name}")
            
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
    
    else:
        print("⚠️  NO GPU DETECTED")
        print("Training will use CPU (slower but functional)")
        print("\nTo enable GPU support:")
        print("1. Update your GPU drivers")
        print("2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print("3. Install cuDNN: https://developer.nvidia.com/cudnn")
    
    # Test performance
    print("\n" + "=" * 30)
    print("🚀 PERFORMANCE TEST")
    print("=" * 30)
    
    device_name = '/GPU:0' if gpus else '/CPU:0'
    device_type = 'GPU' if gpus else 'CPU'
    
    with tf.device(device_name):
        # Matrix multiplication test
        import time
        
        print(f"Testing {device_type} performance...")
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        
        start = time.time()
        c = tf.matmul(a, b)
        end = time.time()
        
        print(f"✅ Matrix multiplication on {device_type}: {(end-start)*1000:.2f}ms")
    
    # Summary
    print("\n" + "=" * 30)
    print("📋 SUMMARY")
    print("=" * 30)
    
    if gpus:
        print("🎯 READY FOR GPU TRAINING!")
        print("   - GPU acceleration: ENABLED")
        print("   - Mixed precision: ENABLED")
        print("   - Memory growth: ENABLED")
    else:
        print("🐌 CPU TRAINING MODE")
        print("   - Will work but slower")
        print("   - Consider GPU setup for faster training")
    
    return len(gpus) > 0

if __name__ == "__main__":
    gpu_available = check_gpu_setup()
    
    print(f"\n🏁 Setup complete! GPU available: {gpu_available}")
    print("Ready to start MENINGIOMA tumor detection training!")