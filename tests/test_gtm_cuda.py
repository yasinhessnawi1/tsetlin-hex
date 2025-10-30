#!/usr/bin/env python3
"""
Test Graph Tsetlin Machine CUDA initialization.
This script specifically tests the GTM library's CUDA usage.
"""

import sys
import os

print("=" * 60)
print("GRAPH TSETLIN MACHINE CUDA TEST")
print("=" * 60)

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Check CUDA environment
print("\n1. Environment:")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Import GraphTsetlinMachine
print("\n2. Importing GraphTsetlinMachine...")
try:
    from GraphTsetlinMachine.graphs import Graphs
    from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
    print("   ✓ GraphTsetlinMachine imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import GraphTsetlinMachine: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check if PyCUDA initialized
print("\n3. Checking PyCUDA initialization...")
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    device = pycuda.autoinit.device
    print(f"   ✓ PyCUDA initialized")
    print(f"   - Device: {device.name()}")
    print(f"   - Compute Capability: {device.compute_capability()}")
    print(f"   - Total Memory: {device.total_memory() / (1024**3):.2f} GB")
    
    free_mem, total_mem = cuda.mem_get_info()
    print(f"   - Free Memory: {free_mem / (1024**3):.2f} GB")
    
except Exception as e:
    print(f"   ✗ PyCUDA initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create a minimal test graph
print("\n4. Creating test graphs...")
try:
    import numpy as np
    
    # Create 10 simple graphs
    num_graphs = 10
    graphs = Graphs(
        number_of_graphs=num_graphs,
        symbols=['A', 'B', 'C'],
        hypervector_size=128,
        hypervector_bits=4
    )
    
    # Add simple graphs (1-2 nodes each)
    for i in range(num_graphs):
        graph_id = graphs.set_number_of_graph_nodes(2)
        graphs.add_graph_node('A', graph_id, 0)
        graphs.add_graph_node('B' if i % 2 == 0 else 'C', graph_id, 1)
        graphs.add_graph_node_edge(graph_id, 0, 1, edge_type_name="connects")
    
    graphs.encode()
    print(f"   ✓ Created {num_graphs} test graphs")
    
except Exception as e:
    print(f"   ✗ Failed to create graphs: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create GTM model
print("\n5. Creating Graph Tsetlin Machine...")
try:
    labels = np.array([i % 2 for i in range(num_graphs)], dtype=np.int32)
    
    gtm = MultiClassGraphTsetlinMachine(
        number_of_clauses=50,
        T=100,
        s=2.0,
        depth=2,
        message_size=128,
        message_bits=2,
        max_included_literals=32,
        grid=(208, 1, 1),
        block=(128, 1, 1)
    )
    
    print("   ✓ GTM model created")
    
except Exception as e:
    print(f"   ✗ Failed to create GTM: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test training
print("\n6. Testing training (1 epoch)...")
try:
    print("   Starting fit...")
    gtm.fit(graphs, labels, epochs=1, incremental=False)
    print("   ✓ Training completed successfully")
    
    # Check GPU memory usage
    free_mem_after, total_mem_after = cuda.mem_get_info()
    used_mem = (total_mem - free_mem_after) / (1024**2)
    print(f"   - GPU Memory Used: {used_mem:.2f} MB")
    
except Exception as e:
    print(f"   ✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test prediction
print("\n7. Testing prediction...")
try:
    predictions = gtm.predict(graphs)
    accuracy = 100.0 * np.sum(predictions == labels) / len(labels)
    print(f"   ✓ Prediction successful")
    print(f"   - Accuracy: {accuracy:.2f}%")
    
except Exception as e:
    print(f"   ✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL GTM CUDA TESTS PASSED!")
print("=" * 60)
print("\nThe Graph Tsetlin Machine is successfully using the GPU.")
print()

