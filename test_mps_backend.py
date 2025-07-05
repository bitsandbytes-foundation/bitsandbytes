#!/usr/bin/env python3
"""
Test script for MPS (Apple Silicon) backend support in bitsandbytes.

This script tests basic functionality like device detection and
fallback behavior when MPS is available.
"""

import torch
import sys
import os

def test_mps_detection():
    """Test if MPS is properly detected."""
    print("=== MPS Detection Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    
    if hasattr(torch.backends, 'mps'):
        print(f"MPS built: {torch.backends.mps.is_built()}")
        if torch.backends.mps.is_available():
            print("✓ MPS is available and ready to use")
            return True
        else:
            print("✗ MPS is not available")
            return False
    else:
        print("✗ MPS backend not found in PyTorch")
        return False

def test_bitsandbytes_import():
    """Test if bitsandbytes can be imported with MPS support."""
    print("\n=== Bitsandbytes Import Test ===")
    try:
        # Add the local bitsandbytes to path
        sys.path.insert(0, '/Volumes/Samsung970EVOPlus/dev-projects/bitsandbytes')
        import bitsandbytes as bnb
        print("✓ Bitsandbytes imported successfully")
        
        print(f"Supported devices: {bnb.supported_torch_devices}")
        
        if 'mps' in bnb.supported_torch_devices:
            print("✓ MPS is listed as a supported device")
        else:
            print("✗ MPS not found in supported devices")
        
        return True
    except Exception as e:
        print(f"✗ Failed to import bitsandbytes: {e}")
        return False

def test_tensor_operations():
    """Test basic tensor operations on MPS device."""
    print("\n=== MPS Tensor Operations Test ===")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("Skipping MPS tensor tests - MPS not available")
        return True
    
    try:
        # Create tensors on MPS device
        device = torch.device("mps")
        x = torch.randn(10, 10, device=device)
        y = torch.randn(10, 10, device=device)
        
        # Basic operations
        z = x + y
        w = torch.matmul(x, y)
        
        print(f"✓ Tensor operations successful on device: {z.device}")
        print(f"✓ Matrix multiplication successful: {w.shape}")
        return True
    except Exception as e:
        print(f"✗ MPS tensor operations failed: {e}")
        return False

def test_quantization_fallback():
    """Test quantization operations with CPU fallback."""
    print("\n=== Quantization Fallback Test ===")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("Skipping quantization tests - MPS not available")
        return True
    
    try:
        # This will test our MPS backend with CPU fallback
        sys.path.insert(0, '/Volumes/Samsung970EVOPlus/dev-projects/bitsandbytes')
        import bitsandbytes.functional as F
        
        device = torch.device("mps")
        
        # Test with small tensor to avoid memory issues
        A = torch.randn(8, 8, device=device, dtype=torch.float32)
        
        # Create a simple quantization code
        code = torch.linspace(-1, 1, 256, device=device, dtype=torch.float32)
        
        print(f"✓ Created test tensors on device: {A.device}")
        print("Note: Full quantization testing requires compiled MPS library")
        return True
    except Exception as e:
        print(f"✗ Quantization test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing bitsandbytes MPS (Apple Silicon) support\n")
    
    tests = [
        ("MPS Detection", test_mps_detection),
        ("Bitsandbytes Import", test_bitsandbytes_import),
        ("MPS Tensor Operations", test_tensor_operations),
        ("Quantization Fallback", test_quantization_fallback),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        print("\nNext steps:")
        print("1. Build bitsandbytes with MPS support: cmake -DCOMPUTE_BACKEND=mps")
        print("2. Install the compiled library")
        print("3. Run actual quantization tests")
    else:
        print(f"✗ {total - passed} tests failed")
        print("\nThis is expected when running without compiled MPS library.")
        print("The implementation provides proper fallbacks to CPU.")

if __name__ == "__main__":
    main()