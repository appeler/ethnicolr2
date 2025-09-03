#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive test to verify all ethnicolr2 functions work with sample names.
This test should be run before and after any code changes to ensure functionality.
"""

import pandas as pd
import sys
import os

# Add the package to path for testing
sys.path.insert(0, '/Users/soodoku/Documents/GitHub/ethnicolr2')

try:
    from ethnicolr2 import (
        pred_fl_full_name, 
        pred_fl_last_name, 
        pred_census_last_name,
        census_ln
    )
    print("✓ Successfully imported all ethnicolr2 functions")
except ImportError as e:
    print(f"✗ Failed to import ethnicolr2 functions: {e}")
    sys.exit(1)

def test_sample_names():
    """Test all functions with a diverse set of sample names"""
    
    # Test data with names that should predict to different ethnicities
    test_names = [
        {"first": "John", "last": "Smith", "expected_category": "nh_white"},
        {"first": "Maria", "last": "Garcia", "expected_category": "hispanic"}, 
        {"first": "Wei", "last": "Zhang", "expected_category": "asian"},
        {"first": "Jamal", "last": "Johnson", "expected_category": "nh_black"},
        {"first": "David", "last": "Kim", "expected_category": "asian"},
        {"first": "Sarah", "last": "Williams", "expected_category": "nh_white"},
    ]
    
    df = pd.DataFrame(test_names)
    print(f"\nTesting with {len(df)} sample names:")
    for _, row in df.iterrows():
        print(f"  - {row['first']} {row['last']} (expected: {row['expected_category']})")
    
    print("\n" + "="*60)
    
    # Test 1: Florida Last Name Model
    print("1. Testing Florida Last Name Model (pred_fl_last_name)...")
    try:
        result_fl_ln = pred_fl_last_name(df.copy(), lname_col='last')
        print("✓ Florida Last Name predictions:")
        for i, row in result_fl_ln.iterrows():
            probs = row['probs']
            max_prob = max(probs.values())
            print(f"   {row['last']:10} -> {row['preds']:10} (confidence: {max_prob:.3f})")
        print(f"   Result shape: {result_fl_ln.shape}")
        print(f"   Columns: {list(result_fl_ln.columns)}")
    except Exception as e:
        print(f"✗ Florida Last Name model failed: {e}")
        return False
    
    print("\n" + "-"*40)
    
    # Test 2: Florida Full Name Model  
    print("2. Testing Florida Full Name Model (pred_fl_full_name)...")
    try:
        result_fl_fn = pred_fl_full_name(df.copy(), lname_col='last', fname_col='first')
        print("✓ Florida Full Name predictions:")
        for i, row in result_fl_fn.iterrows():
            probs = row['probs']
            max_prob = max(probs.values())
            print(f"   {row['first']} {row['last']:10} -> {row['preds']:10} (confidence: {max_prob:.3f})")
        print(f"   Result shape: {result_fl_fn.shape}")
        print(f"   Columns: {list(result_fl_fn.columns)}")
    except Exception as e:
        print(f"✗ Florida Full Name model failed: {e}")
        return False
        
    print("\n" + "-"*40)
    
    # Test 3: Census Last Name Model
    print("3. Testing Census Last Name Model (pred_census_last_name)...")
    try:
        result_cen_ln = pred_census_last_name(df.copy(), lname_col='last')
        print("✓ Census Last Name predictions:")
        for i, row in result_cen_ln.iterrows():
            probs = row['probs']
            max_prob = max(probs.values())
            print(f"   {row['last']:10} -> {row['preds']:10} (confidence: {max_prob:.3f})")
        print(f"   Result shape: {result_cen_ln.shape}")
        print(f"   Columns: {list(result_cen_ln.columns)}")
    except Exception as e:
        print(f"✗ Census Last Name model failed: {e}")
        return False
        
    print("\n" + "-"*40)
    
    # Test 4: Census Lookup (not ML prediction)
    print("4. Testing Census Lookup (census_ln)...")
    try:
        result_census_2010 = census_ln(df.copy(), lname_col='last', year=2010)
        print("✓ Census 2010 lookup:")
        for i, row in result_census_2010.iterrows():
            pct_white = row.get('pctwhite', 'N/A')
            pct_hispanic = row.get('pcthispanic', 'N/A')
            print(f"   {row['last']:10} -> White: {pct_white}%, Hispanic: {pct_hispanic}%")
        print(f"   Result shape: {result_census_2010.shape}")
        print(f"   Columns: {list(result_census_2010.columns)}")
    except Exception as e:
        print(f"✗ Census lookup failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ All functions completed successfully!")
    return True

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n5. Testing Edge Cases...")
    
    # Test empty dataframe
    try:
        empty_df = pd.DataFrame(columns=['last'])
        result = pred_fl_last_name(empty_df, 'last')
        print(f"✓ Empty dataframe handled: {len(result)} rows returned")
    except Exception as e:
        print(f"✓ Empty dataframe properly rejected: {e}")
    
    # Test missing column
    try:
        df = pd.DataFrame([{"wrong_col": "smith"}])
        result = pred_fl_last_name(df, 'last')
        print("✗ Missing column should have failed")
    except Exception as e:
        print(f"✓ Missing column properly caught: {e}")
    
    # Test special characters
    try:
        df = pd.DataFrame([{"last": "O'Connor"}, {"last": "García-López"}])
        result = pred_fl_last_name(df, 'last')
        print(f"✓ Special characters handled: {len(result)} predictions")
    except Exception as e:
        print(f"? Special characters issue: {e}")

if __name__ == "__main__":
    print("Testing ethnicolr2 functionality with sample names...")
    print("="*60)
    
    success = test_sample_names()
    test_edge_cases()
    
    if success:
        print("\n🎉 All core functionality verified!")
        print("Ready to proceed with code improvements.")
    else:
        print("\n❌ Some tests failed. Need to fix issues before improvements.")
        sys.exit(1)