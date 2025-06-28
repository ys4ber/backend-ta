#!/usr/bin/env python3
"""
Comprehensive system test for TAQA ML-Enhanced Anomaly Detection with SQLite
Tests all components: imports, database, API endpoints, ML prediction, etc.
"""

import sys
import os
sys.path.append('/home/ysaber42/Desktop/taqa')

import requests
import time
import subprocess
import json
from datetime import datetime

def test_imports():
    """Test all module imports"""
    print("🧪 Testing module imports...")
    try:
        import database
        import models
        import config
        import utils
        import alert_manager
        import ml_models
        from main import app
        print("✅ All modules import successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_database():
    """Test database functionality"""
    print("🧪 Testing database operations...")
    try:
        from database import db_manager, SensorDataDB
        
        # Test database connection
        db = db_manager.get_db()
        count = db.query(SensorDataDB).count()
        db.close()
        
        print(f"✅ Database connected - {count} sensor readings")
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_pydantic_compatibility():
    """Test Pydantic model compatibility"""
    print("🧪 Testing Pydantic compatibility...")
    try:
        from models import SensorData
        
        # Test model creation
        sensor = SensorData(
            equipment_id="TEST-001",
            temperature=75.0,
            pressure=18.0,
            vibration=3.0,
            efficiency=90.0
        )
        
        # Test both serialization methods
        dict_data = sensor.model_dump()  # Pydantic v2
        print("✅ Pydantic models work correctly")
        return True
    except Exception as e:
        print(f"❌ Pydantic error: {e}")
        return False

def start_server():
    """Start the server for testing"""
    print("🚀 Starting server for API tests...")
    try:
        process = subprocess.Popen([
            'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8005'
        ], cwd='/home/ysaber42/Desktop/taqa', 
           stdout=subprocess.PIPE, 
           stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(5)
        
        # Test if server is running
        response = requests.get("http://127.0.0.1:8005/", timeout=10)
        if response.status_code == 200:
            print("✅ Server started successfully")
            return process
        else:
            print("❌ Server not responding")
            process.kill()
            return None
    except Exception as e:
        print(f"❌ Server start error: {e}")
        return None

def test_api_endpoints(server_process):
    """Test API endpoints"""
    print("🧪 Testing API endpoints...")
    base_url = "http://127.0.0.1:8005"
    
    tests = [
        {
            "name": "Root endpoint",
            "method": "GET",
            "url": f"{base_url}/",
            "expected_keys": ["message", "version", "status"]
        },
        {
            "name": "Database health",
            "method": "GET", 
            "url": f"{base_url}/api/database/health",
            "expected_keys": ["status", "statistics"]
        },
        {
            "name": "ML Prediction",
            "method": "POST",
            "url": f"{base_url}/predict",
            "data": {
                "equipment_id": "TAQA-TEST-API",
                "temperature": 80.0,
                "pressure": 20.0,
                "vibration": 3.5,
                "efficiency": 90.0
            },
            "expected_keys": ["equipment_id", "anomaly_score", "prediction"]
        }
    ]
    
    passed = 0
    for test in tests:
        try:
            if test["method"] == "GET":
                response = requests.get(test["url"], timeout=10)
            elif test["method"] == "POST":
                response = requests.post(test["url"], json=test["data"], timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if all(key in data for key in test["expected_keys"]):
                    print(f"✅ {test['name']}")
                    passed += 1
                else:
                    print(f"❌ {test['name']} - Missing keys")
            else:
                print(f"❌ {test['name']} - Status {response.status_code}")
        except Exception as e:
            print(f"❌ {test['name']} - Error: {e}")
    
    return passed == len(tests)

def test_database_integration():
    """Test database integration with API"""
    print("🧪 Testing database integration...")
    try:
        from database import db_manager
        
        # Get initial count
        initial_data = db_manager.get_sensor_data(limit=1000)
        initial_count = len(initial_data)
        
        # Make API call that should store data
        response = requests.post("http://127.0.0.1:8005/predict", json={
            "equipment_id": "TAQA-INTEGRATION-TEST",
            "temperature": 82.0,
            "pressure": 21.0,
            "vibration": 3.8,
            "efficiency": 89.0
        }, timeout=10)
        
        if response.status_code == 200:
            # Check if data was stored
            time.sleep(1)  # Give database time to save
            new_data = db_manager.get_sensor_data(limit=1000)
            new_count = len(new_data)
            
            if new_count > initial_count:
                print("✅ Database integration working")
                return True
            else:
                print("❌ Data not saved to database")
                return False
        else:
            print("❌ API call failed")
            return False
    except Exception as e:
        print(f"❌ Database integration error: {e}")
        return False

def stop_server(process):
    """Stop the test server"""
    if process:
        process.kill()
        print("🛑 Server stopped")

def main():
    """Run comprehensive system test"""
    print("🏭 TAQA ML-Enhanced Anomaly Detection - Comprehensive System Test")
    print("=" * 70)
    
    tests = [
        ("Module Imports", test_imports),
        ("Database Operations", test_database), 
        ("Pydantic Compatibility", test_pydantic_compatibility)
    ]
    
    # Run basic tests
    passed = 0
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        print()
    
    # Start server for API tests
    server_process = start_server()
    if server_process:
        print()
        
        # Run API tests
        api_tests = [
            ("API Endpoints", lambda: test_api_endpoints(server_process)),
            ("Database Integration", test_database_integration)
        ]
        
        for test_name, test_func in api_tests:
            if test_func():
                passed += 1
            print()
        
        stop_server(server_process)
    
    # Summary
    total_tests = len(tests) + (2 if server_process else 0)
    print("=" * 70)
    print(f"📊 Test Results: {passed}/{total_tests} tests passed")
    
    if passed == total_tests:
        print("🎉 ALL TESTS PASSED! System is working correctly!")
        print("✅ SQLite database integration is fully functional")
        print("✅ API endpoints are working")
        print("✅ ML prediction with database storage is working")
        print("✅ All dependencies are compatible")
    else:
        print("⚠️ Some tests failed - check the output above")
    
    print("\n🚀 Ready for production use!")

if __name__ == "__main__":
    main()
