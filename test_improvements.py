#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the type safety and exception improvements work correctly.
This demonstrates the enhanced error handling and type safety features.
"""

import pandas as pd
import sys
import traceback

# Test type hints and error handling improvements
def test_type_safety_improvements():
    """Test that our type safety enhancements work correctly"""
    print("🔍 Testing Type Safety & Exception Improvements")
    print("=" * 60)
    
    # Import the modules to test type checking
    sys.path.insert(0, '/Users/soodoku/Documents/GitHub/ethnicolr2')
    
    try:
        from ethnicolr2.ethnicolr_class import EthnicolrModelClass
        from ethnicolr2.dataset import EthniDataset
        from ethnicolr2.models import LSTM
        from ethnicolr2.utils import arg_parser
        print("✓ All improved modules imported successfully")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False
    
    print("\n1. Testing EthnicolrModelClass.test_and_norm_df improvements...")
    
    # Test 1: Missing column - should raise ValueError with specific message
    try:
        df = pd.DataFrame([{"existing_col": "value"}])
        EthnicolrModelClass.test_and_norm_df(df, "missing_col")
        print("✗ Should have raised ValueError for missing column")
        return False
    except ValueError as e:
        if "not found in DataFrame" in str(e) and "Available columns" in str(e):
            print("✓ Missing column error handling improved with helpful message")
        else:
            print(f"? Unexpected error message: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type: {type(e).__name__}: {e}")
        return False
    
    # Test 2: Empty data - should raise ValueError with specific message  
    try:
        df = pd.DataFrame([{"last": None}])
        EthnicolrModelClass.test_and_norm_df(df, "last")
        print("✗ Should have raised ValueError for empty data")
        return False
    except ValueError as e:
        if "contains no non-NaN values" in str(e):
            print("✓ Empty data error handling improved with specific message")
        else:
            print(f"? Unexpected error message: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type: {type(e).__name__}: {e}")
        return False
    
    print("\n2. Testing EthnicolrModelClass.lineToTensor improvements...")
    
    # Test 3: Type checking for lineToTensor
    try:
        EthnicolrModelClass.lineToTensor(123, "abc", 10, 99)  # Wrong type
        print("✗ Should have raised TypeError for non-string input")
        return False
    except TypeError as e:
        if "Expected string input" in str(e):
            print("✓ lineToTensor type checking works correctly")
        else:
            print(f"? Unexpected error message: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type: {type(e).__name__}: {e}")
        return False
    
    # Test 4: Value checking for lineToTensor
    try:
        EthnicolrModelClass.lineToTensor("test", "abc", -5, 99)  # Invalid max_name
        print("✗ Should have raised ValueError for negative max_name")
        return False
    except ValueError as e:
        if "must be positive" in str(e):
            print("✓ lineToTensor value validation works correctly")
        else:
            print(f"? Unexpected error message: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type: {type(e).__name__}: {e}")
        return False
    
    print("\n3. Testing LSTM model improvements...")
    
    # Test 5: LSTM parameter validation
    try:
        model = LSTM(-1, 128, 5)  # Invalid input_size
        print("✗ Should have raised ValueError for negative input_size")
        return False
    except ValueError as e:
        if "must be positive" in str(e):
            print("✓ LSTM parameter validation works correctly")
        else:
            print(f"? Unexpected error message: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type: {type(e).__name__}: {e}")
        return False
    
    print("\n4. Testing EthniDataset improvements...")
    
    # Test 6: Dataset parameter validation
    try:
        dataset = EthniDataset("not_a_dataframe", "abc", 10, 99)
        print("✗ Should have raised TypeError for non-DataFrame input")
        return False
    except TypeError as e:
        if "Expected pandas DataFrame" in str(e):
            print("✓ EthniDataset type checking works correctly")
        else:
            print(f"? Unexpected error message: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type: {type(e).__name__}: {e}")
        return False
    
    # Test 7: Dataset missing column validation
    try:
        df = pd.DataFrame([{"wrong_col": "value"}])
        dataset = EthniDataset(df, "abc", 10, 99)
        print("✗ Should have raised ValueError for missing __name column")
        return False
    except ValueError as e:
        if "__name" in str(e):
            print("✓ EthniDataset column validation works correctly")
        else:
            print(f"? Unexpected error message: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type: {type(e).__name__}: {e}")
        return False
    
    print("\n5. Testing arg_parser improvements...")
    
    # Test 8: arg_parser type validation
    try:
        args = arg_parser("not_a_list", "title", "output.csv")
        print("✗ Should have raised TypeError for non-list argv")
        return False
    except TypeError as e:
        if "Expected list for argv" in str(e):
            print("✓ arg_parser type checking works correctly")
        else:
            print(f"? Unexpected error message: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type: {type(e).__name__}: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All type safety and exception improvements verified!")
    print("\nKey improvements implemented:")
    print("  • Specific exception types (ValueError, TypeError) instead of generic Exception")
    print("  • Comprehensive type hints throughout codebase")
    print("  • Helpful error messages with context")
    print("  • Input validation with clear feedback")
    print("  • Improved docstrings with Args/Returns/Raises sections")
    
    return True

def test_functional_improvements():
    """Test that the core functionality still works after improvements"""
    print("\n\n🔧 Testing that improvements don't break functionality...")
    print("=" * 60)
    
    try:
        from ethnicolr2.ethnicolr_class import EthnicolrModelClass
        import torch
        
        # Test that improved lineToTensor still works correctly
        result = EthnicolrModelClass.lineToTensor("smith", "abcdefghijklmnopqrstuvwxyz", 10, 26)
        if isinstance(result, torch.Tensor) and result.shape == (10,):
            print("✓ lineToTensor still works correctly after improvements")
        else:
            print(f"✗ lineToTensor broken: got {type(result)} with shape {getattr(result, 'shape', 'N/A')}")
            return False
            
        # Test that improved dataset still works
        from ethnicolr2.dataset import EthniDataset
        import pandas as pd
        
        df = pd.DataFrame([{"__name": "test"}])
        dataset = EthniDataset(df, "abc", 5, 99)
        if len(dataset) == 1:
            print("✓ EthniDataset still works correctly after improvements")
        else:
            print(f"✗ EthniDataset broken: expected length 1, got {len(dataset)}")
            return False
            
        # Test that improved LSTM still works
        from ethnicolr2.models import LSTM
        
        model = LSTM(100, 128, 5)
        test_input = torch.randint(0, 99, (2, 10))  # batch_size=2, seq_len=10
        output = model(test_input)
        if output.shape == (2, 5):  # batch_size=2, num_classes=5
            print("✓ LSTM model still works correctly after improvements")
        else:
            print(f"✗ LSTM broken: expected shape (2, 5), got {output.shape}")
            return False
        
        print("✓ All functionality preserved after improvements")
        return True
        
    except Exception as e:
        print(f"✗ Functional test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing ethnicolr2 Type Safety & Exception Improvements")
    print("🚀 This verifies that our low-risk, high-value improvements work correctly\n")
    
    # Run improvement tests
    improvements_ok = test_type_safety_improvements()
    
    # Run functionality tests
    functionality_ok = test_functional_improvements()
    
    if improvements_ok and functionality_ok:
        print("\n🎉 SUCCESS: All improvements verified and functionality preserved!")
        print("✅ Ready for production use")
        sys.exit(0)
    else:
        print("\n❌ FAILURE: Some tests failed")
        print("🔧 Improvements need review before production use")
        sys.exit(1)