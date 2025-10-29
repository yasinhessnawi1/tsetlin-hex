#!/usr/bin/env python3
"""
Setup script to detect and configure CUDA for MIG devices.
Run this before training to ensure proper GPU usage.
"""

import os
import sys
import subprocess

def run_command(cmd):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def main():
    print("=" * 70)
    print("MIG CUDA SETUP FOR GRAPH TSETLIN MACHINE")
    print("=" * 70)
    
    # Check nvidia-smi
    print("\n1. Checking NVIDIA driver and GPUs...")
    stdout, stderr, rc = run_command("nvidia-smi")
    if rc != 0:
        print("   ✗ nvidia-smi failed. Is NVIDIA driver installed?")
        print(f"   Error: {stderr}")
        sys.exit(1)
    
    print("   ✓ NVIDIA driver detected")
    
    # Check for MIG mode
    print("\n2. Checking for MIG devices...")
    stdout, stderr, rc = run_command("nvidia-smi -L")
    
    mig_devices = []
    gpu_devices = []
    
    for line in stdout.split('\n'):
        if line.strip():
            if 'MIG' in line:
                # Extract MIG device info
                # Format: "MIG 0g.20gb Device 0: (UUID: MIG-...)"
                mig_devices.append(line.strip())
            elif 'GPU' in line:
                gpu_devices.append(line.strip())
    
    print(f"   Found {len(gpu_devices)} GPU(s)")
    print(f"   Found {len(mig_devices)} MIG device(s)")
    
    if mig_devices:
        print("\n   MIG Devices:")
        for i, dev in enumerate(mig_devices):
            print(f"     [{i}] {dev}")
    
    # Get current CUDA_VISIBLE_DEVICES
    current_cvd = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"\n3. Current CUDA_VISIBLE_DEVICES: {current_cvd}")
    
    # Recommend settings
    print("\n4. Recommendations:")
    print("   " + "-" * 66)
    
    if mig_devices:
        print("   Your system has MIG mode enabled.")
        print("   ")
        print("   For PyCUDA to work with MIG devices, you need to:")
        print("   ")
        print("   Option 1: Use the base GPU device (recommended for PyCUDA)")
        print("     export CUDA_VISIBLE_DEVICES=0")
        print("   ")
        print("   Option 2: If Option 1 doesn't work, disable MIG mode:")
        print("     sudo nvidia-smi -mig 0  # Disable MIG on GPU 0")
        print("     sudo nvidia-smi -pm 1   # Enable persistence mode")
        print("   ")
        print("   Note: PyCUDA's pycuda.autoinit may have issues with MIG.")
        print("   The codebase uses pycuda.autoinit which doesn't fully")
        print("   support MIG device UUIDs.")
    else:
        print("   No MIG devices detected. Standard GPU configuration.")
        if len(gpu_devices) > 0:
            print(f"   You can use: export CUDA_VISIBLE_DEVICES=0")
    
    print("   " + "-" * 66)
    
    # Test PyCUDA
    print("\n5. Testing PyCUDA import...")
    try:
        # Set device if not set
        if current_cvd == 'Not set':
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            print("   Setting CUDA_VISIBLE_DEVICES=0 for test...")
        
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        device = pycuda.autoinit.device
        print("   ✓ PyCUDA initialized successfully!")
        print(f"   - Device: {device.name()}")
        print(f"   - Compute Capability: {device.compute_capability()}")
        print(f"   - Total Memory: {device.total_memory() / (1024**3):.2f} GB")
        
        free_mem, total_mem = cuda.mem_get_info()
        print(f"   - Free Memory: {free_mem / (1024**3):.2f} GB")
        
    except Exception as e:
        print(f"   ✗ PyCUDA initialization failed!")
        print(f"   Error: {e}")
        print("\n   This is likely because:")
        print("   - MIG mode is enabled and PyCUDA can't access it properly")
        print("   - You need to either disable MIG or use the base GPU device")
        return 1
    
    print("\n" + "=" * 70)
    print("✓ SETUP COMPLETE")
    print("=" * 70)
    print("\nTo run training with GPU:")
    print("  export CUDA_VISIBLE_DEVICES=0")
    print("  ./run_strong_training.sh")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

