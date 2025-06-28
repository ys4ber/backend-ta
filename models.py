# models.py - Pydantic Data Models
"""
Data models for the TAQA ML-Enhanced Anomaly Detection System
All request/response models with validation using Pydantic v2
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

# Input Models
class SensorData(BaseModel):
    """Input model for sensor readings"""
    equipment_id: str = Field(..., description="Unique equipment identifier")
    temperature: float = Field(..., ge=-50, le=200, description="Temperature in Celsius")
    pressure: float = Field(..., ge=0, le=100, description="Pressure in bar")
    vibration: float = Field(..., ge=0, le=20, description="Vibration in mm/s")
    efficiency: float = Field(..., ge=0, le=100, description="Efficiency percentage")
    timestamp: Optional[str] = Field(None, description="Timestamp of reading")
    location: Optional[str] = Field("TAQA-FACILITY", description="Equipment location")

    class Config:
        json_schema_extra = {
            "example": {
                "equipment_id": "TAQA-POMPE-001",
                "temperature": 75.5,
                "pressure": 18.2,
                "vibration": 3.1,
                "efficiency": 92.5,
                "timestamp": "2025-06-27T10:30:00",
                "location": "TAQA-FACILITY"
            }
        }

# Alert Models
class Alert(BaseModel):
    """Alert model for equipment issues"""
    id: str = Field(..., description="Unique alert identifier")
    equipment_id: str = Field(..., description="Equipment that triggered alert")
    equipment_type: str = Field(..., description="Type of equipment")
    severity: str = Field(..., description="Alert severity level")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Detailed alert message")
    triggered_at: str = Field(..., description="When alert was triggered")
    sensor_data: Dict[str, float] = Field(..., description="Sensor readings that triggered alert")
    recommended_actions: List[str] = Field(..., description="Recommended maintenance actions")
    estimated_cost: Optional[float] = Field(None, description="Estimated maintenance cost")

# Predictive Analysis Model
class PredictiveAnalysis(BaseModel):
    """Predictive maintenance analysis results"""
    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_type: str = Field(..., description="Equipment type")
    current_health_score: float = Field(..., ge=0, le=100, description="Current health score (0-100)")
    predicted_failure_probability: float = Field(..., ge=0, le=1, description="Failure probability (0-1)")
    estimated_remaining_life: str = Field(..., description="Estimated remaining useful life")
    maintenance_window: str = Field(..., description="Recommended maintenance window")
    cost_analysis: Dict[str, float] = Field(..., description="Cost breakdown analysis")
    recommendations: List[str] = Field(..., description="Maintenance recommendations")

# Main ML Prediction Model
class MLAnomalyPrediction(BaseModel):
    """Complete ML anomaly prediction response"""
    equipment_id: str = Field(..., description="Equipment identifier")
    anomaly_score: float = Field(..., ge=0, le=100, description="Anomaly score (0-100)")
    prediction: str = Field(..., description="Prediction category (CRITICAL, HIGH, MEDIUM, LOW, NORMAL)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    severity: str = Field(..., description="Severity in French")
    recommendation: str = Field(..., description="Quick recommendation summary")
    sensor_readings: Dict[str, float] = Field(..., description="Input sensor readings")
    timestamp: str = Field(..., description="Prediction timestamp")
    ml_analysis: Dict[str, Any] = Field(..., description="ML algorithm results")
    rule_analysis: Dict[str, str] = Field(..., description="Rule-based analysis results")
    alerts_triggered: List[Alert] = Field(..., description="Alerts generated from this prediction")
    predictive_analysis: PredictiveAnalysis = Field(..., description="Predictive maintenance analysis")
    maintenance_priority: int = Field(..., ge=1, le=5, description="Maintenance priority (1=highest)")
    algorithm_used: str = Field(..., description="Algorithm used for prediction")

# Excel Analysis Models
class ExcelAnomalyResult(BaseModel):
    """Result for a single equipment from Excel analysis"""
    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_type: str = Field(..., description="Equipment type")
    description: str = Field(..., description="Anomaly description from Excel")
    section: str = Field(..., description="Equipment section/department")
    detection_date: str = Field(..., description="Date anomaly was detected")
    status: str = Field(..., description="Current status")
    priority: int = Field(..., ge=1, le=5, description="Priority level")
    ai_prediction: MLAnomalyPrediction = Field(..., description="AI/ML prediction results")
    urgency_level: str = Field(..., description="Calculated urgency level")
    urgency_color: str = Field(..., description="Color code for urgency")
    urgency_action: str = Field(..., description="Recommended action based on urgency")
    sensor_data: Dict[str, float] = Field(..., description="Generated sensor data")

class ExcelAnalysisResponse(BaseModel):
    """Complete Excel file analysis response"""
    summary: Dict[str, Dict[str, int]] = Field(..., description="Summary by equipment type")
    detailed_results: List[ExcelAnomalyResult] = Field(..., description="Detailed results for each equipment")
    total_processed: int = Field(..., description="Total number of records processed")
    anomalies_detected: int = Field(..., description="Number of anomalies detected")
    processing_time: float = Field(..., description="Processing time in seconds")
    filename: str = Field(..., description="Original filename")
    algorithm_used: str = Field(..., description="Algorithm used for analysis")
    ml_performance: Dict[str, Any] = Field(..., description="ML model performance metrics")

# Database Response Models
class SensorDataResponse(BaseModel):
    """Response model for sensor data from database"""
    id: int = Field(..., description="Database record ID")
    equipment_id: str = Field(..., description="Equipment identifier")
    temperature: float = Field(..., description="Temperature reading")
    pressure: float = Field(..., description="Pressure reading")
    vibration: float = Field(..., description="Vibration reading")
    efficiency: float = Field(..., description="Efficiency reading")
    location: str = Field(..., description="Equipment location")
    timestamp: str = Field(..., description="Reading timestamp")
    created_at: str = Field(..., description="Record creation timestamp")

class AlertResponse(BaseModel):
    """Response model for alerts from database"""
    id: int = Field(..., description="Database record ID")
    alert_id: str = Field(..., description="Unique alert identifier")
    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_type: str = Field(..., description="Equipment type")
    severity: str = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    sensor_data: Dict[str, float] = Field(..., description="Sensor readings")
    recommended_actions: List[str] = Field(..., description="Recommended actions")
    estimated_cost: Optional[float] = Field(None, description="Estimated cost")
    triggered_at: str = Field(..., description="Alert trigger timestamp")
    resolved_at: Optional[str] = Field(None, description="Alert resolution timestamp")
    is_active: bool = Field(..., description="Alert active status")
    created_at: str = Field(..., description="Record creation timestamp")

class PredictionResponse(BaseModel):
    """Response model for predictions from database"""
    id: int = Field(..., description="Database record ID")
    equipment_id: str = Field(..., description="Equipment identifier")
    anomaly_score: float = Field(..., description="Anomaly score")
    prediction: str = Field(..., description="Prediction category")
    confidence: float = Field(..., description="Prediction confidence")
    severity: str = Field(..., description="Severity level")
    recommendation: str = Field(..., description="Recommendation")
    algorithm_used: str = Field(..., description="Algorithm used")
    maintenance_priority: int = Field(..., description="Maintenance priority")
    sensor_readings: Dict[str, float] = Field(..., description="Sensor readings")
    ml_analysis: Dict[str, Any] = Field(..., description="ML analysis results")
    rule_analysis: Dict[str, str] = Field(..., description="Rule analysis results")
    timestamp: str = Field(..., description="Prediction timestamp")
    created_at: str = Field(..., description="Record creation timestamp")

class EquipmentStatsResponse(BaseModel):
    """Response model for equipment statistics"""
    equipment_id: str = Field(..., description="Equipment identifier")
    sensor_data_count: int = Field(..., description="Number of sensor readings")
    alerts_count: int = Field(..., description="Total number of alerts")
    active_alerts_count: int = Field(..., description="Number of active alerts")
    predictions_count: int = Field(..., description="Number of predictions")
    latest_sensor_reading: Optional[Dict[str, Any]] = Field(None, description="Latest sensor reading")
    latest_prediction: Optional[Dict[str, Any]] = Field(None, description="Latest prediction")
    latest_alert: Optional[Dict[str, Any]] = Field(None, description="Latest alert")

class DatabaseCleanupResponse(BaseModel):
    """Response model for database cleanup operation"""
    sensor_data_deleted: int = Field(..., description="Number of sensor records deleted")
    alerts_deleted: int = Field(..., description="Number of alert records deleted")
    predictions_deleted: int = Field(..., description="Number of prediction records deleted")
    cutoff_date: str = Field(..., description="Cutoff date for cleanup")

# Request Models for Database Operations
class AlertResolveRequest(BaseModel):
    """Request model for resolving an alert"""
    alert_id: str = Field(..., description="Alert ID to resolve")

class DatabaseQueryRequest(BaseModel):
    """Request model for database queries"""
    equipment_id: Optional[str] = Field(None, description="Equipment ID filter")
    limit: Optional[int] = Field(100, ge=1, le=1000, description="Maximum number of records to return")
    active_only: Optional[bool] = Field(True, description="For alerts: return only active alerts")
