#!/usr/bin/env python3
"""
Test script for TAQA SQLite database integration
Run this to verify database functionality
"""

import os
import sys
sys.path.append('/home/ysaber42/Desktop/taqa')

from database import db_manager
from datetime import datetime
import json

def test_database():
    """Test all database operations"""
    print("🔧 Testing TAQA SQLite Database Integration")
    print("=" * 50)
    
    # Test 1: Database initialization
    print("\n1. Testing database initialization...")
    try:
        db_manager.create_tables()
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    
    # Test 2: Save sensor data
    print("\n2. Testing sensor data saving...")
    try:
        test_sensor = {
            'equipment_id': 'TAQA-TEST-001',
            'temperature': 75.5,
            'pressure': 18.2,
            'vibration': 3.1,
            'efficiency': 92.5,
            'location': 'TAQA-FACILITY',
            'timestamp': datetime.now().isoformat()
        }
        
        sensor_id = db_manager.save_sensor_data(test_sensor)
        print(f"✅ Sensor data saved with ID: {sensor_id}")
    except Exception as e:
        print(f"❌ Sensor data saving failed: {e}")
        return False
    
    # Test 3: Save alert
    print("\n3. Testing alert saving...")
    try:
        test_alert = {
            'id': f'test_alert_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'equipment_id': 'TAQA-TEST-001',
            'equipment_type': 'POMPE',
            'severity': 'HIGH',
            'title': 'Test Alert',
            'message': 'This is a test alert',
            'sensor_data': {'temperature': 75.5, 'pressure': 18.2},
            'recommended_actions': ['Check temperature', 'Inspect pump'],
            'estimated_cost': 5000.0,
            'triggered_at': datetime.now().isoformat()
        }
        
        alert_id = db_manager.save_alert(test_alert)
        print(f"✅ Alert saved with ID: {alert_id}")
    except Exception as e:
        print(f"❌ Alert saving failed: {e}")
        return False
    
    # Test 4: Save prediction
    print("\n4. Testing prediction saving...")
    try:
        test_prediction = {
            'equipment_id': 'TAQA-TEST-001',
            'anomaly_score': 85.5,
            'prediction': 'HIGH',
            'confidence': 0.92,
            'severity': 'Élevé',
            'recommendation': 'Maintenance requise',
            'algorithm_used': 'Random Forest',
            'maintenance_priority': 2,
            'sensor_readings': {'temperature': 75.5, 'pressure': 18.2},
            'ml_analysis': {'model': 'RandomForest', 'score': 0.92},
            'rule_analysis': {'temperature': 'HIGH', 'pressure': 'NORMAL'},
            'timestamp': datetime.now().isoformat()
        }
        
        prediction_id = db_manager.save_prediction(test_prediction)
        print(f"✅ Prediction saved with ID: {prediction_id}")
    except Exception as e:
        print(f"❌ Prediction saving failed: {e}")
        return False
    
    # Test 5: Retrieve data
    print("\n5. Testing data retrieval...")
    try:
        # Get sensor data
        sensor_data = db_manager.get_sensor_data(equipment_id='TAQA-TEST-001', limit=10)
        print(f"✅ Retrieved {len(sensor_data)} sensor records")
        
        # Get alerts
        alerts = db_manager.get_alerts(equipment_id='TAQA-TEST-001', limit=10)
        print(f"✅ Retrieved {len(alerts)} alert records")
        
        # Get predictions
        predictions = db_manager.get_predictions(equipment_id='TAQA-TEST-001', limit=10)
        print(f"✅ Retrieved {len(predictions)} prediction records")
        
        # Get equipment stats
        stats = db_manager.get_equipment_stats('TAQA-TEST-001')
        print(f"✅ Retrieved equipment stats: {stats['sensor_data_count']} sensors, {stats['alerts_count']} alerts")
        
    except Exception as e:
        print(f"❌ Data retrieval failed: {e}")
        return False
    
    # Test 6: Database health
    print("\n6. Testing database health...")
    try:
        from database import SensorDataDB, AlertDB, PredictionDB
        db = db_manager.get_db()
        
        sensor_count = db.query(SensorDataDB).count()
        alert_count = db.query(AlertDB).count()
        prediction_count = db.query(PredictionDB).count()
        
        db.close()
        
        print(f"✅ Database health check passed:")
        print(f"   - Sensor readings: {sensor_count}")
        print(f"   - Alerts: {alert_count}")
        print(f"   - Predictions: {prediction_count}")
        
    except Exception as e:
        print(f"❌ Database health check failed: {e}")
        return False
    
    # Test 7: Check database file
    db_file = "/home/ysaber42/Desktop/taqa/taqa_anomaly_detection.db"
    if os.path.exists(db_file):
        size_mb = os.path.getsize(db_file) / (1024 * 1024)
        print(f"\n✅ Database file created: {db_file}")
        print(f"   Size: {size_mb:.2f} MB")
    else:
        print(f"\n❌ Database file not found: {db_file}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All database tests passed successfully!")
    print("💾 SQLite database is ready for the TAQA system")
    print("🚀 You can now run the main application with database support")
    
    return True

if __name__ == "__main__":
    test_database()
