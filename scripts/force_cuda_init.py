#!/usr/bin/env python3
"""
Force CUDA initialization before importing GraphTsetlinMachine.
This ensures the correct GPU device is used with MIG mode.

Import this module BEFORE importing GraphTsetlinMachine to ensure
proper CUDA device selection.

Usage:
    import sys
    import os
    sys.path.insert(0, 'scripts')
    import force_cuda_init  # Must be before GraphTsetlinMachine import
    
    from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
"""

import os
import sys

# Ensure CUDA device is set before any CUDA imports
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    print("WARNING: CUDA_VISIBLE_DEVICES not set. Setting to 0...")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print(f"CUDA Device Selection: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

# Force PyCUDA to initialize with the selected device
try:
    import pycuda.driver as cuda
    
    # Initialize CUDA driver
    cuda.init()
    
    # Get the device
    device_num = 0
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cvd = os.environ['CUDA_VISIBLE_DEVICES']
        if cvd and cvd.isdigit():
            device_num = 0  # After CUDA_VISIBLE_DEVICES is set, device 0 is the selected device
    
    device = cuda.Device(device_num)
    
    print(f"CUDA Device Initialized: {device.name()}")
    print(f"  - Compute Capability: {device.compute_capability()}")
    print(f"  - Total Memory: {device.total_memory() / (1024**3):.2f} GB")
    
    # Create context (this will be used by pycuda.autoinit)
    # Note: We don't keep a reference, as pycuda.autoinit will create its own
    
except Exception as e:
    print(f"WARNING: Could not initialize CUDA device: {e}")
    print("Training may not use GPU!")

