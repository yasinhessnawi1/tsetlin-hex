#!/usr/bin/env python3
"""
Test CUDA availability and PyCUDA setup.
"""

import sys
import os

print("=" * 60)
print("CUDA AVAILABILITY TEST")
print("=" * 60)

# Check environment
print("\n1. Environment Variables:")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"   CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"   CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")

# Try importing PyCUDA
print("\n2. Testing PyCUDA import...")
try:
    import pycuda
    print(f"   ✓ PyCUDA version: {pycuda.VERSION_TEXT}")
except ImportError as e:
    print(f"   ✗ PyCUDA import failed: {e}")
    sys.exit(1)

# Try PyCUDA driver
print("\n3. Testing PyCUDA driver...")
try:
    import pycuda.driver as cuda
    print("   ✓ PyCUDA driver imported")
except ImportError as e:
    print(f"   ✗ PyCUDA driver import failed: {e}")
    sys.exit(1)

# Try CUDA initialization
print("\n4. Testing CUDA initialization...")
try:
    import pycuda.autoinit
    print("   ✓ CUDA auto-initialized")
    
    # Get device info
    device = pycuda.autoinit.device
    print(f"\n   Device Information:")
    print(f"   - Name: {device.name()}")
    print(f"   - Compute Capability: {device.compute_capability()}")
    print(f"   - Total Memory: {device.total_memory() / (1024**3):.2f} GB")
    print(f"   - Multiprocessors: {device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)}")
    
    # Get context info
    context = pycuda.autoinit.context
    free_mem, total_mem = cuda.mem_get_info()
    print(f"\n   Memory Information:")
    print(f"   - Free: {free_mem / (1024**3):.2f} GB")
    print(f"   - Total: {total_mem / (1024**3):.2f} GB")
    print(f"   - Used: {(total_mem - free_mem) / (1024**3):.2f} GB")
    
except Exception as e:
    print(f"   ✗ CUDA initialization failed: {e}")
    print(f"\n   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try a simple CUDA operation
print("\n5. Testing CUDA operations...")
try:
    import numpy as np
    import pycuda.gpuarray as gpuarray
    
    # Create array on GPU
    test_array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    gpu_array = gpuarray.to_gpu(test_array)
    result = gpu_array.get()
    
    if np.allclose(test_array, result):
        print("   ✓ GPU memory allocation and transfer working")
    else:
        print("   ✗ GPU operation produced incorrect results")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ CUDA operation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try compiling a simple kernel
print("\n6. Testing CUDA kernel compilation...")
try:
    from pycuda.compiler import SourceModule
    
    mod = SourceModule("""
    __global__ void test_kernel(float *dest, float *a, float *b)
    {
      const int i = threadIdx.x;
      dest[i] = a[i] + b[i];
    }
    """)
    
    test_kernel = mod.get_function("test_kernel")
    print("   ✓ CUDA kernel compiled successfully")
    
    # Test the kernel
    a = np.random.randn(400).astype(np.float32)
    b = np.random.randn(400).astype(np.float32)
    
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    dest_gpu = gpuarray.empty_like(a_gpu)
    
    test_kernel(dest_gpu, a_gpu, b_gpu, block=(400, 1, 1), grid=(1, 1))
    
    result = dest_gpu.get()
    expected = a + b
    
    if np.allclose(result, expected):
        print("   ✓ CUDA kernel execution successful")
    else:
        print("   ✗ CUDA kernel produced incorrect results")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ CUDA kernel compilation/execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL CUDA TESTS PASSED!")
print("=" * 60)
print("\nYour system is ready for GPU-accelerated training.")
print()

