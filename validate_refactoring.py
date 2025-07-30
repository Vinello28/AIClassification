#!/usr/bin/env python
"""
Validation script to test the refactored AI Classification system
"""
import sys
import os

def test_imports():
    """Test that all imports work correctly"""
    print("🧪 Testing imports...")
    
    try:
        # Test config import (should work without torch)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from ai_classification.core.config import CATEGORIES
        print(f"✅ Config loaded: {len(CATEGORIES)} categories")
        
        # Test training data import (may need torch but shouldn't fail)
        try:
            from ai_classification.data.training_data import ALL_TRAINING_DATA
            print(f"✅ Training data loaded: {len(ALL_TRAINING_DATA)} samples")
        except ImportError as e:
            print(f"⚠️  Training data import failed (expected without torch): {e}")
        
        # Test package imports
        import ai_classification
        print(f"✅ Package imported: version {ai_classification.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_backward_compatibility():
    """Test that backward compatibility wrappers work"""
    print("\n🔄 Testing backward compatibility...")
    
    try:
        # Test old-style imports
        import ai_classifier
        print("✅ ai_classifier wrapper works")
        
        import client  
        print("✅ client wrapper works")
        
        import server
        print("✅ server wrapper works")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False

def test_docker_readiness():
    """Test that Docker requirements are met"""
    print("\n🐳 Testing Docker readiness...")
    
    required_files = [
        'docker/Dockerfile',
        'docker/docker-compose.yml', 
        'docker/nginx.conf',
        '.dockerignore',
        'requirements.txt',
        'src/ai_classification/api/server.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing Docker files: {missing_files}")
        return False
    else:
        print("✅ All Docker configuration files present")
        return True

def test_structure():
    """Test that the directory structure is correct"""  
    print("\n📁 Testing directory structure...")
    
    expected_structure = [
        'src/ai_classification',
        'src/ai_classification/core',
        'src/ai_classification/api', 
        'src/ai_classification/data',
        'src/ai_classification/utils',
        'tests',
        'scripts',
        'docker'
    ]
    
    missing_dirs = []
    for directory in expected_structure:
        if not os.path.isdir(directory):
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ Directory structure is correct")
        return True

def main():
    """Run all validation tests"""
    print("🚀 AI Classification Refactoring Validation")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_structure),
        ("Imports", test_imports),
        ("Backward Compatibility", test_backward_compatibility), 
        ("Docker Readiness", test_docker_readiness)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n🎯 Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactoring was successful.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())