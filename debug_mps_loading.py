#!/usr/bin/env python3
"""
Debug MPS library loading.
"""

import torch
import sys
import os

# Add the local bitsandbytes to path
sys.path.insert(0, '/Volumes/Samsung970EVOPlus/dev-projects/bitsandbytes')

def debug_library_loading():
    """Debug the library loading process."""
    print("🔍 DEBUGGING MPS LIBRARY LOADING")
    print("=" * 50)
    
    print("1️⃣ Environment Check:")
    print(f"   MPS Available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    print(f"   MPS Built: {torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False}")
    
    print("\n2️⃣ Library Files Check:")
    mps_lib = "/Volumes/Samsung970EVOPlus/dev-projects/bitsandbytes/bitsandbytes/libbitsandbytes_mps.dylib"
    cpu_lib = "/Volumes/Samsung970EVOPlus/dev-projects/bitsandbytes/bitsandbytes/libbitsandbytes_cpu.dylib"
    
    print(f"   MPS library exists: {os.path.exists(mps_lib)}")
    print(f"   CPU library exists: {os.path.exists(cpu_lib)}")
    
    if os.path.exists(mps_lib):
        stat = os.stat(mps_lib)
        print(f"   MPS library size: {stat.st_size} bytes")
    
    print("\n3️⃣ Import Process:")
    try:
        import bitsandbytes.cextension as ce
        print(f"   BNB_BACKEND: {ce.BNB_BACKEND}")
        print(f"   HIP_ENVIRONMENT: {ce.HIP_ENVIRONMENT}")
        
        # Check library loading
        print(f"   Library object type: {type(ce.lib)}")
        
        if hasattr(ce.lib, 'compiled_with_cuda'):
            print(f"   Compiled with CUDA: {ce.lib.compiled_with_cuda}")
        
        # Try to access an MPS function
        if hasattr(ce.lib, 'quantize_blockwise_mps'):
            print("   ✅ quantize_blockwise_mps function found!")
        else:
            print("   ❌ quantize_blockwise_mps function NOT found")
            
        # Check all available functions
        print(f"\n4️⃣ Available functions (first 10):")
        if hasattr(ce.lib, '_lib'):
            lib_funcs = [attr for attr in dir(ce.lib._lib) if not attr.startswith('_')][:10]
            for func in lib_funcs:
                print(f"   - {func}")
        else:
            print("   No _lib attribute found")
            
    except Exception as e:
        print(f"   Import error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n5️⃣ CUDA Specs Check:")
    try:
        from bitsandbytes.cuda_specs import get_cuda_specs
        cuda_specs = get_cuda_specs()
        print(f"   CUDA specs: {cuda_specs}")
    except Exception as e:
        print(f"   CUDA specs error: {e}")

if __name__ == "__main__":
    debug_library_loading()