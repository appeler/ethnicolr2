#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic functionality test - validates core prediction functions work with sample names.
This can be run in environments where dependencies are properly installed.
"""

import pandas as pd
import sys
import os

def test_basic_predictions():
    """Test core prediction functions with sample names"""
    print("🧪 Testing Basic Prediction Functionality")
    print("=" * 50)
    
    # Set up path
    sys.path.insert(0, '/Users/soodoku/Documents/GitHub/ethnicolr2')
    
    # Create test data
    test_names = pd.DataFrame([
        {"first": "John", "last": "Smith"},
        {"first": "Maria", "last": "Garcia"}, 
        {"first": "Wei", "last": "Zhang"},
    ])
    
    print("Test data:")
    print(test_names.to_string(index=False))
    print()
    
    try:
        print("Importing ethnicolr2...")
        from ethnicolr2 import (
            pred_fl_full_name,
            pred_fl_last_name, 
            pred_census_last_name,
            census_ln
        )
        print("✓ Import successful")
        
        print("\nTesting Florida Last Name predictions...")
        fl_ln_result = pred_fl_last_name(test_names.copy(), 'last')
        print("✓ Florida Last Name model working")
        for _, row in fl_ln_result.iterrows():
            print(f"  {row['last']:8} -> {row['preds']}")
        
        print("\nTesting Florida Full Name predictions...")
        fl_fn_result = pred_fl_full_name(test_names.copy(), lname_col='last', fname_col='first')
        print("✓ Florida Full Name model working")
        for _, row in fl_fn_result.iterrows():
            print(f"  {row['first']} {row['last']:8} -> {row['preds']}")
        
        print("\nTesting Census Last Name predictions...")
        cen_ln_result = pred_census_last_name(test_names.copy(), 'last')
        print("✓ Census Last Name model working")
        for _, row in cen_ln_result.iterrows():
            print(f"  {row['last']:8} -> {row['preds']}")
        
        print("\nTesting Census lookup...")
        census_result = census_ln(test_names.copy(), 'last', 2010)
        print("✓ Census lookup working")
        for _, row in census_result.iterrows():
            white_pct = row.get('pctwhite', 'N/A')
            print(f"  {row['last']:8} -> {white_pct}% white")
        
        print("\n✅ All basic functionality tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed - dependencies not installed: {e}")
        print("\nTo install dependencies:")
        print("  python -m venv venv")
        print("  source venv/bin/activate")
        print("  pip install -e '.[test]'")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_predictions()
    sys.exit(0 if success else 1)