#!/usr/bin/env python3
"""
Test the build system and compilation readiness for MPS backend.
"""

import subprocess
import os
import sys

def test_cmake_availability():
    """Test if cmake is available and supports our configuration."""
    print("🔨 Testing CMake Availability...")
    try:
        result = subprocess.run(['cmake', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ {version}")
            return True
        else:
            print("❌ CMake not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ CMake not found or not responding")
        return False

def test_cmake_mps_configuration():
    """Test if CMake can configure with MPS backend."""
    print("\n🔧 Testing CMake MPS Configuration...")
    
    # Change to project directory
    original_cwd = os.getcwd()
    project_root = '/Volumes/Samsung970EVOPlus/dev-projects/bitsandbytes'
    
    try:
        os.chdir(project_root)
        
        # Test cmake configuration (dry run)
        cmd = ['cmake', '-DCOMPUTE_BACKEND=mps', '-B', 'test_build_mps', '.']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ CMake MPS configuration successful")
            print("✅ Build system ready for MPS compilation")
            
            # Clean up test build directory
            subprocess.run(['rm', '-rf', 'test_build_mps'], capture_output=True)
            return True
        else:
            print("❌ CMake MPS configuration failed")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ CMake configuration timed out")
        return False
    except Exception as e:
        print(f"❌ CMake test failed: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def test_mps_source_files():
    """Test that all required MPS source files are present."""
    print("\n📁 Testing MPS Source Files...")
    
    project_root = '/Volumes/Samsung970EVOPlus/dev-projects/bitsandbytes'
    required_files = [
        'csrc/mps_ops.mm',
        'csrc/mps_ops.h',
        'csrc/mps_kernels.metal',
        'bitsandbytes/backends/mps/ops.py',
        'bitsandbytes/backends/mps/__init__.py'
    ]
    
    all_present = True
    for file_path in required_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_present = False
    
    return all_present

def test_cmake_mps_variables():
    """Test that CMakeLists.txt has proper MPS support."""
    print("\n⚙️ Testing CMakeLists.txt MPS Support...")
    
    cmake_file = '/Volumes/Samsung970EVOPlus/dev-projects/bitsandbytes/CMakeLists.txt'
    
    try:
        with open(cmake_file, 'r') as f:
            content = f.read()
        
        checks = [
            ('COMPUTE_BACKEND.*mps', 'MPS backend option'),
            ('BUILD_MPS', 'BUILD_MPS variable'),
            ('MPS_FILES', 'MPS source files'),
            ('mps_ops.mm', 'MPS Objective-C++ source'),
            ('mps_kernels.metal', 'Metal kernels')
        ]
        
        import re
        all_found = True
        for pattern, description in checks:
            if re.search(pattern, content, re.IGNORECASE):
                print(f"✅ {description}")
            else:
                print(f"❌ {description} - NOT FOUND")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ Error reading CMakeLists.txt: {e}")
        return False

def test_xcode_availability():
    """Test if Xcode tools are available for Metal compilation."""
    print("\n🍎 Testing Xcode/Metal Tools...")
    
    try:
        # Check xcode-select
        result = subprocess.run(['xcode-select', '-p'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Xcode path: {result.stdout.strip()}")
        else:
            print("❌ Xcode command line tools not properly installed")
            return False
        
        # Check for Metal compiler
        result = subprocess.run(['xcrun', '-f', 'metal'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Metal compiler available: {result.stdout.strip()}")
            return True
        else:
            print("❌ Metal compiler not found")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Xcode tools not available")
        return False

def main():
    print("🏗️  MPS BUILD SYSTEM VALIDATION")
    print("=" * 50)
    
    tests = [
        ("CMake Availability", test_cmake_availability),
        ("MPS Source Files", test_mps_source_files),
        ("CMakeLists.txt MPS Support", test_cmake_mps_variables),
        ("Xcode/Metal Tools", test_xcode_availability),
        ("CMake MPS Configuration", test_cmake_mps_configuration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*len(test_name)}")
        print(test_name)
        print(f"{'='*len(test_name)}")
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🏗️  BUILD SYSTEM TEST RESULTS")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 BUILD SYSTEM: FULLY READY!")
        print("✅ You can now compile MPS support with:")
        print("   cmake -DCOMPUTE_BACKEND=mps -B build .")
        print("   cmake --build build")
    elif passed >= 4:
        print("✅ BUILD SYSTEM: MOSTLY READY")
        print("Minor issues detected - should still be buildable")
    elif passed >= 3:
        print("⚠️  BUILD SYSTEM: NEEDS ATTENTION")
        print("Some components missing or misconfigured")
    else:
        print("🚨 BUILD SYSTEM: NOT READY")
        print("Major configuration issues detected")

if __name__ == "__main__":
    main()