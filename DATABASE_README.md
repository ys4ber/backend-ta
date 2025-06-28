# TAQA Anomaly Detection System - SQLite Database Integration

## Overview

The TAQA ML-Enhanced Anomaly Detection System now includes comprehensive SQLite database support for persistent data storage and retrieval. This integration allows you to store sensor data, alerts, ML predictions, and analysis results for historical tracking and analytics.

## Database Features

### 🗄️ **Database Tables**
- **sensor_data**: Stores all sensor readings from equipment
- **alerts**: Tracks equipment alerts with severity levels and resolution status
- **predictions**: Stores ML model predictions and confidence scores
- **predictive_analysis**: Saves predictive maintenance analysis results
- **excel_analysis**: Archives Excel file analysis results

### 📊 **Automatic Data Persistence**
- All sensor data is automatically saved when using `/predict` endpoint
- ML predictions are stored with complete analysis details
- Alerts are tracked with timestamps and resolution status
- Excel file analysis results are archived for future reference

## New API Endpoints

### 📖 **Data Retrieval Endpoints**

#### Get Sensor Data
```http
GET /api/sensor-data?equipment_id=TAQA-POMPE-001&limit=100
```
Returns historical sensor readings for equipment.

#### Get Alerts
```http
GET /api/alerts?equipment_id=TAQA-POMPE-001&active_only=true&limit=100
```
Returns alerts with filtering options.

#### Get Predictions
```http
GET /api/predictions?equipment_id=TAQA-POMPE-001&limit=100
```
Returns ML prediction history.

#### Get Equipment Statistics
```http
GET /api/equipment/{equipment_id}/stats
```
Returns comprehensive statistics for specific equipment including:
- Total sensor readings count
- Total and active alerts count
- Predictions count
- Latest readings and analysis

### 🔧 **Management Endpoints**

#### Resolve Alert
```http
POST /api/alerts/{alert_id}/resolve
```
Marks an alert as resolved.

#### Database Health Check
```http
GET /api/database/health
```
Returns database connectivity status and table statistics.

#### Database Cleanup
```http
DELETE /api/cleanup?days_to_keep=90
```
Removes old data (older than specified days) to maintain performance.

## Database Schema

### Sensor Data Table
```sql
CREATE TABLE sensor_data (
    id INTEGER PRIMARY KEY,
    equipment_id VARCHAR NOT NULL,
    temperature FLOAT NOT NULL,
    pressure FLOAT NOT NULL,
    vibration FLOAT NOT NULL,
    efficiency FLOAT NOT NULL,
    location VARCHAR DEFAULT 'TAQA-FACILITY',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Alerts Table
```sql
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    alert_id VARCHAR UNIQUE NOT NULL,
    equipment_id VARCHAR NOT NULL,
    equipment_type VARCHAR NOT NULL,
    severity VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    message TEXT NOT NULL,
    sensor_data JSON,
    recommended_actions JSON,
    estimated_cost FLOAT,
    triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolved_at DATETIME NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Predictions Table
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    equipment_id VARCHAR NOT NULL,
    anomaly_score FLOAT NOT NULL,
    prediction VARCHAR NOT NULL,
    confidence FLOAT NOT NULL,
    severity VARCHAR NOT NULL,
    recommendation TEXT NOT NULL,
    algorithm_used VARCHAR NOT NULL,
    maintenance_priority INTEGER NOT NULL,
    sensor_readings JSON,
    ml_analysis JSON,
    rule_analysis JSON,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Usage Examples

### 1. Making a Prediction (with automatic database storage)
```python
import requests

# Send sensor data for prediction
response = requests.post("http://localhost:8000/predict", json={
    "equipment_id": "TAQA-POMPE-001",
    "temperature": 85.5,
    "pressure": 22.3,
    "vibration": 4.2,
    "efficiency": 88.5,
    "location": "TAQA-FACILITY"
})

# Data is automatically stored in database
prediction = response.json()
print(f"Prediction: {prediction['prediction']} (Score: {prediction['anomaly_score']})")
```

### 2. Retrieving Historical Data
```python
# Get sensor data history
response = requests.get("http://localhost:8000/api/sensor-data?equipment_id=TAQA-POMPE-001")
sensor_history = response.json()

# Get equipment statistics
response = requests.get("http://localhost:8000/api/equipment/TAQA-POMPE-001/stats")
stats = response.json()
print(f"Total readings: {stats['sensor_data_count']}")
print(f"Active alerts: {stats['active_alerts_count']}")
```

### 3. Alert Management
```python
# Get active alerts
response = requests.get("http://localhost:8000/api/alerts?active_only=true")
alerts = response.json()

# Resolve an alert
requests.post("http://localhost:8000/api/alerts/alert_20250627_123456_TAQA-POMPE-001/resolve")
```

## Database File Location

The SQLite database file is created at:
```
/home/ysaber42/Desktop/taqa/taqa_anomaly_detection.db
```

You can access this file directly using SQLite tools for advanced queries and analysis.

## Benefits

### 📈 **Analytics and Reporting**
- Track equipment performance trends over time
- Analyze prediction accuracy and model performance
- Generate maintenance reports based on historical data

### 🔍 **Historical Analysis**
- Compare current readings with historical baselines
- Identify recurring issues and failure patterns
- Monitor equipment degradation over time

### 🚨 **Alert Management**
- Track alert resolution times
- Analyze alert frequency by equipment type
- Monitor system effectiveness

### 💾 **Data Persistence**
- No data loss during system restarts
- Complete audit trail of all predictions and alerts
- Backup and restore capabilities

## Testing

A comprehensive test script is included to verify database functionality:

```bash
cd /home/ysaber42/Desktop/taqa
source venv/bin/activate
python test_database.py
```

This script tests:
- Database table creation
- Data insertion (sensor data, alerts, predictions)
- Data retrieval and querying
- Database health checks
- File system integration

## API Documentation

Full API documentation with interactive testing is available at:
```
http://localhost:8000/docs
```

After starting the server, you can explore all endpoints, view request/response schemas, and test the database integration directly through the web interface.

## Next Steps

1. **Start the server**: `uvicorn main:app --host 0.0.0.0 --port 8000`
2. **Access the API docs**: Visit `http://localhost:8000/docs`
3. **Test predictions**: Use the `/predict` endpoint with sensor data
4. **View stored data**: Check the new database endpoints
5. **Monitor equipment**: Use the statistics endpoint for insights

The system now provides a complete solution for ML-enhanced anomaly detection with persistent data storage, making it suitable for production deployments and long-term monitoring scenarios.
