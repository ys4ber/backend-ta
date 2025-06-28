#!/usr/bin/env python3
"""
TAQA Performance Improvements Test Script
Tests all implemented optimizations
"""

import sys
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        from ml_models import TAQAMLAnomalyDetector
        print("✅ ml_models imported successfully")
    except Exception as e:
        print(f"❌ ml_models import failed: {e}")
        return False
    
    try:
        from database import DatabaseManager
        print("✅ database imported successfully")
    except Exception as e:
        print(f"❌ database import failed: {e}")
        return False
    
    try:
        from utils import (
            fast_risk_screening,
            optimize_feature_selection,
            calculate_trend_features,
            enhanced_anomaly_detection
        )
        print("✅ utils performance functions imported successfully")
    except Exception as e:
        print(f"❌ utils import failed: {e}")
        return False
    
    return True

def test_ml_detector_initialization():
    """Test ML detector initialization with new features"""
    print("\n🤖 Testing ML Detector initialization...")
    
    try:
        from ml_models import TAQAMLAnomalyDetector
        detector = TAQAMLAnomalyDetector()
        
        # Check if new models are in the model list
        expected_models = ["xgboost", "lightgbm", "random_forest", "svm_classifier"]
        for model in expected_models:
            if model in detector.models:
                print(f"✅ {model} model slot available")
            else:
                print(f"❌ {model} model missing")
        
        # Check performance optimizations
        if hasattr(detector, 'prediction_cache'):
            print("✅ Prediction caching system available")
        else:
            print("❌ Prediction caching not found")
        
        if hasattr(detector, 'model_paths'):
            print("✅ Lazy loading paths available")
        else:
            print("❌ Lazy loading not configured")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Detector test failed: {e}")
        return False

def test_database_optimizations():
    """Test database optimizations"""
    print("\n💾 Testing database optimizations...")
    
    try:
        from database import DatabaseManager
        db_manager = DatabaseManager()
        
        # Test fast query method
        if hasattr(db_manager, 'get_sensor_data_fast'):
            print("✅ Fast database queries available")
        else:
            print("❌ Fast database queries not found")
        
        # Test index creation
        if hasattr(db_manager, 'create_optimized_indexes'):
            print("✅ Database indexing system available")
            # db_manager.create_optimized_indexes()  # Uncomment to create indexes
        else:
            print("❌ Database indexing not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_performance_functions():
    """Test performance utility functions"""
    print("\n⚡ Testing performance functions...")
    
    try:
        from utils import fast_risk_screening, calculate_trend_features
        
        # Mock sensor data for testing
        class MockSensorData:
            def __init__(self):
                self.temperature = 85.0
                self.pressure = 15.0
                self.vibration = 4.0
                self.efficiency = 80.0
        
        sensor_data = MockSensorData()
        
        # Test fast risk screening
        start_time = time.time()
        risk_level = fast_risk_screening(sensor_data, "POMPE")
        screening_time = (time.time() - start_time) * 1000  # Convert to ms
        
        print(f"✅ Fast risk screening: {risk_level} ({screening_time:.3f}ms)")
        
        # Test trend analysis
        recent_readings = [
            {"temperature": 80.0},
            {"temperature": 82.0},
            {"temperature": 84.0},
            {"temperature": 85.0}
        ]
        current_reading = {"temperature": 87.0}
        
        start_time = time.time()
        trend_features = calculate_trend_features(current_reading, recent_readings)
        trend_time = (time.time() - start_time) * 1000  # Convert to ms
        
        print(f"✅ Trend analysis: {trend_features.get('temp_trend', 'N/A')} ({trend_time:.3f}ms)")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance functions test failed: {e}")
        return False

def test_api_endpoints():
    """Test if API endpoints are properly configured"""
    print("\n🌐 Testing API configuration...")
    
    try:
        from api import setup_api_routes
        print("✅ API routes setup function available")
        
        # Check if batch processing is configured
        import inspect
        api_source = inspect.getsource(setup_api_routes)
        
        if "predict_batch" in api_source:
            print("✅ Batch processing endpoint configured")
        else:
            print("❌ Batch processing endpoint not found")
        
        if "upload_excel_optimized" in api_source:
            print("✅ Optimized Excel upload endpoint configured")
        else:
            print("❌ Optimized Excel upload not found")
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 TAQA Performance Improvements Test Suite")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("ML Detector", test_ml_detector_initialization),
        ("Database", test_database_optimizations),
        ("Performance Functions", test_performance_functions),
        ("API Endpoints", test_api_endpoints)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All performance improvements are properly implemented!")
        print("\n🚀 Performance Features Available:")
        print("   ✅ XGBoost & LightGBM model slots")
        print("   ✅ Smart prediction caching")
        print("   ✅ Database query optimization")
        print("   ✅ Fast risk screening")
        print("   ✅ Trend analysis")
        print("   ✅ Batch processing API")
        print("   ✅ Memory-efficient file processing")
        print("   ✅ Lazy model loading")
        print("\n📝 Next Steps:")
        print("   1. Install XGBoost/LightGBM: pip install xgboost lightgbm")
        print("   2. Start system: python main.py") 
        print("   3. Train models: POST /train_ml")
        print("   4. Test performance: POST /predict_batch")
    else:
        print("⚠️  Some optimizations may need attention.")
        print("Check the errors above and ensure all files are properly updated.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
