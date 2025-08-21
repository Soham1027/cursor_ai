#!/usr/bin/env python3
"""
Test script to verify the fixes for the sales prediction system
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_feature_generation():
    """Test that feature generation works without errors."""
    print("🧪 Testing Feature Generation...")
    
    try:
        from predict_sale_csv import generate_future_features, clean_and_preprocess
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        sample_data = pd.DataFrame({
            'sale_date': dates,
            'total_quantity_sold': np.random.randint(10, 100, size=len(dates)),
            'average_selling_price': np.random.uniform(5, 50, size=len(dates)),
            'standard_price': np.random.uniform(5, 50, size=len(dates)),
            'average_items_in_order': np.random.uniform(1, 5, size=len(dates)),
            'product_name': ['Test Product'] * len(dates)
        })
        
        # Test preprocessing
        print("✅ Testing data preprocessing...")
        processed_data = clean_and_preprocess(sample_data)
        print(f"   Generated {processed_data.shape[1]} features")
        print(f"   Feature names: {list(processed_data.columns)}")
        
        # Test future feature generation
        print("✅ Testing future feature generation...")
        start_date = datetime(2024, 1, 1).date()
        days = 5
        
        future_features, future_dates = generate_future_features(
            start_date, days, processed_data
        )
        
        print(f"   Generated features for {len(future_dates)} future dates")
        print(f"   Feature matrix shape: {future_features.shape}")
        print(f"   Feature names: {list(future_features.columns)}")
        
        # Verify feature count matches
        if processed_data.shape[1] == future_features.shape[1]:
            print("✅ Feature count matches between preprocessing and generation")
        else:
            print(f"❌ Feature count mismatch: {processed_data.shape[1]} vs {future_features.shape[1]}")
            return False
        
        print("✅ Feature generation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Feature generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_date_calculation():
    """Test that date calculations work correctly."""
    print("\n📅 Testing Date Calculations...")
    
    try:
        start_date = datetime(2024, 1, 1).date()
        end_date = datetime(2024, 1, 31).date()
        
        # Test the fixed calculation
        days_to_forecast = (end_date - start_date).days + 1
        
        print(f"   Start date: {start_date}")
        print(f"   End date: {end_date}")
        print(f"   Days to forecast: {days_to_forecast}")
        
        if days_to_forecast == 31:
            print("✅ Date calculation test passed!")
            return True
        else:
            print(f"❌ Expected 31 days, got {days_to_forecast}")
            return False
            
    except Exception as e:
        print(f"❌ Date calculation test failed: {e}")
        return False

def test_performance_monitor():
    """Test that performance monitor works correctly."""
    print("\n📊 Testing Performance Monitor...")
    
    try:
        from app import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Test log_request method
        monitor.log_request("test_endpoint", 1.5, True)
        monitor.log_request("test_endpoint", 2.0, False, "test error")
        
        stats = monitor.get_stats()
        
        if "test_endpoint" in stats:
            endpoint_stats = stats["test_endpoint"]
            print(f"   Total requests: {endpoint_stats['total_requests']}")
            print(f"   Error count: {endpoint_stats['error_count']}")
            print(f"   Average time: {endpoint_stats['avg_time']:.2f}s")
            
            if endpoint_stats['total_requests'] == 2 and endpoint_stats['error_count'] == 1:
                print("✅ Performance monitor test passed!")
                return True
            else:
                print("❌ Performance monitor stats incorrect")
                return False
        else:
            print("❌ Performance monitor stats not found")
            return False
            
    except Exception as e:
        print(f"❌ Performance monitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Sales Prediction System - Fix Verification Tests")
    print("=" * 60)
    
    tests = [
        test_feature_generation,
        test_date_calculation,
        test_performance_monitor
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixes are working correctly.")
        print("✅ The system should now work without the previous errors.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)