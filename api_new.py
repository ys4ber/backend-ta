# api.py - TAQA API Endpoints
"""
API endpoints for the TAQA ML-Enhanced Anomaly Detection System
Organized into clear sections for easy understanding and modification

SECTIONS:
1. IMPORTS AND SETUP
2. CORE API ENDPOINTS (System Info, ML Operations)
3. DATA PROCESSING ENDPOINTS (File Upload, Sensor Data)
4. MONITORING ENDPOINTS (Health, Stats, Alerts)
5. DATABASE ENDPOINTS (CRUD Operations)
6. TESTING ENDPOINTS (Simulation, Testing)
7. ROUTE SETUP FUNCTION
"""

# ===================================================================
# 1. IMPORTS AND SETUP
# ===================================================================

from fastapi import HTTPException, UploadFile, File, BackgroundTasks
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import io
import logging
from datetime import datetime

# Import our modular components
from models import (
    SensorData, 
    MLAnomalyPrediction, 
    ExcelAnalysisResponse,
    ExcelAnomalyResult,
    SensorDataResponse,
    AlertResponse,
    PredictionResponse,
    EquipmentStatsResponse
)
from utils import (
    extract_equipment_type_from_description,
    generate_sensor_data_from_description,
    calculate_urgency_level,
    safe_convert_numpy,
    ml_enhanced_anomaly_detection
)
from database import DatabaseManager

# Get logger
logger = logging.getLogger(__name__)

# ===================================================================
# 2. CORE API ENDPOINTS (System Info, ML Operations)
# ===================================================================

def setup_core_endpoints(app, ml_detector, alert_manager):
    """Setup core API endpoints"""
    
    @app.get("/")
    async def root():
        return {
            "message": "🏭 TAQA ML-Enhanced Anomaly Detection System",
            "version": "4.0.2",
            "status": "active",
            "ml_status": "trained" if ml_detector.is_trained else "training_required",
            "algorithms": {
                "supervised": ["Random Forest", "SVM", "Neural Network"],
                "unsupervised": ["Isolation Forest", "One-Class SVM", "LOF"],
                "deep_learning": ["Autoencoder"],
                "ensemble": "Weighted voting with 7 algorithms"
            },
            "equipment_types": list(ml_detector.equipment_profiles.keys()),
            "features": [
                "Real ML Algorithms",
                "Ensemble Predictions", 
                "Smart Alert System",
                "Predictive Analytics",
                "Cost Analysis",
                "COMPLETELY FIXED: All serialization issues resolved"
            ]
        }

    @app.post("/predict", response_model=MLAnomalyPrediction)
    async def predict_anomaly_endpoint(sensor_data: SensorData):
        """ML-enhanced anomaly prediction with database integration"""
        try:
            # Initialize database manager
            db_manager = DatabaseManager()
            
            # Save sensor data to database
            sensor_data_dict = sensor_data.model_dump()
            sensor_id = db_manager.save_sensor_data(sensor_data_dict)
            logger.info(f"💾 Sensor data saved to database with ID: {sensor_id}")
            
            # Generate ML prediction
            prediction = ml_enhanced_anomaly_detection(sensor_data, ml_detector, alert_manager, logger)
            
            # Save prediction to database
            prediction_dict = prediction.model_dump()
            prediction_id = db_manager.save_prediction(prediction_dict)
            logger.info(f"💾 Prediction saved to database with ID: {prediction_id}")
            
            # Save alerts to database
            for alert in prediction.alerts_triggered:
                alert_dict = alert.model_dump()
                alert_id = db_manager.save_alert(alert_dict)
                logger.info(f"💾 Alert saved to database with ID: {alert_id}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    @app.post("/train_ml")
    async def train_ml_endpoint(background_tasks: BackgroundTasks):
        """Train ML models in background"""
        try:
            background_tasks.add_task(ml_detector.train_models)
            return {
                "message": "🤖 ML training started in background",
                "status": "training_initiated",
                "algorithms": list(ml_detector.algorithms.keys()),
                "training_time_estimate": "2-5 minutes",
                "note": "Real ML algorithms with scikit-learn"
            }
        except Exception as e:
            logger.error(f"❌ Training error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
    
    @app.get("/ml_status")
    async def ml_status():
        """Get ML model status"""
        return {"trained": ml_detector.is_trained, "algorithms": list(ml_detector.algorithms.keys())}

# ===================================================================
# 3. DATA PROCESSING ENDPOINTS (File Upload, Sensor Data)
# ===================================================================

def setup_data_processing_endpoints(app, ml_detector, alert_manager):
    """Setup data processing endpoints"""
    
    @app.post("/upload_excel", response_model=ExcelAnalysisResponse)
    async def upload_excel_analysis(file: UploadFile = File(...)):
        """Upload and analyze Excel file with ML-enhanced detection"""
        start_time = datetime.now()
        
        logger.info(f"🔍 Starting ML-enhanced Excel analysis: {file.filename}")
        
        if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(status_code=400, detail="Only Excel/CSV files allowed")
        
        try:
            # Read file content
            content = await file.read()
            
            # Read with pandas
            if file.filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(content))
            else:
                df = pd.read_excel(io.BytesIO(content))
            
            logger.info(f"📊 File loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Process each row with ML detection
            anomaly_results = []
            total_anomalies = 0
            high_severity_count = 0
            
            for idx, row in df.iterrows():
                try:
                    # Extract equipment info from description if available
                    description = str(row.get('Description', f'Equipment_{idx}'))
                    equipment_type = extract_equipment_type_from_description(description)
                    
                    # Generate sensor data from the row
                    sensor_data = generate_sensor_data_from_description(row, description, equipment_type)
                    
                    # ML-enhanced prediction
                    prediction = ml_enhanced_anomaly_detection(sensor_data, ml_detector, alert_manager, logger)
                    
                    # Create result
                    result = ExcelAnomalyResult(
                        row_index=idx,
                        equipment_id=sensor_data.equipment_id,
                        equipment_type=equipment_type,
                        is_anomaly=prediction.is_anomaly,
                        anomaly_score=prediction.anomaly_score,
                        severity=prediction.severity,
                        confidence=prediction.confidence,
                        alerts_count=len(prediction.alerts_triggered),
                        cost_impact=prediction.cost_analysis.estimated_cost_impact,
                        maintenance_urgency=prediction.predictive_analysis.maintenance_urgency,
                        failure_probability=prediction.predictive_analysis.failure_probability,
                        description=description
                    )
                    
                    anomaly_results.append(result)
                    
                    if result.is_anomaly:
                        total_anomalies += 1
                        if result.severity == "HIGH":
                            high_severity_count += 1
                            
                except Exception as row_error:
                    logger.warning(f"⚠️ Error processing row {idx}: {str(row_error)}")
                    continue
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create response
            response = ExcelAnalysisResponse(
                filename=file.filename,
                total_rows=len(df),
                anomaly_results=anomaly_results,
                summary={
                    "total_anomalies": total_anomalies,
                    "high_severity_anomalies": high_severity_count,
                    "anomaly_rate": (total_anomalies / len(df)) * 100,
                    "processing_time_seconds": processing_time,
                    "ml_algorithms_used": list(ml_detector.algorithms.keys()),
                    "equipment_types_detected": list(set(r.equipment_type for r in anomaly_results))
                }
            )
            
            logger.info(f"✅ Excel analysis completed: {total_anomalies}/{len(df)} anomalies in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Excel processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ===================================================================
# 4. MONITORING ENDPOINTS (Health, Stats, Alerts)
# ===================================================================

def setup_monitoring_endpoints(app, ml_detector, alert_manager):
    """Setup monitoring endpoints"""
    
    @app.get("/health")
    async def health_check():
        """Comprehensive health check"""
        try:
            db_manager = DatabaseManager()
            db_health = db_manager.health_check()
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "ml_detector": {
                        "status": "ready" if ml_detector.is_trained else "training_required",
                        "algorithms": list(ml_detector.algorithms.keys()),
                        "equipment_profiles": len(ml_detector.equipment_profiles)
                    },
                    "alert_manager": {
                        "status": "active",
                        "active_alerts": len(alert_manager.active_alerts)
                    },
                    "database": db_health
                },
                "version": "4.0.2"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    @app.get("/model_stats")
    async def get_model_statistics():
        """Get detailed ML model statistics"""
        try:
            stats = ml_detector.get_model_info()
            return safe_convert_numpy(stats)
        except Exception as e:
            logger.error(f"❌ Error getting model stats: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")
    
    @app.get("/alerts/active")
    async def get_active_alerts():
        """Get all active alerts"""
        try:
            return {
                "active_alerts": alert_manager.active_alerts,
                "count": len(alert_manager.active_alerts),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error getting alerts: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Alerts error: {str(e)}")

# ===================================================================
# 5. DATABASE ENDPOINTS (CRUD Operations)
# ===================================================================

def setup_database_endpoints(app, ml_detector, alert_manager):
    """Setup database endpoints"""
    
    @app.get("/api/sensor-data", response_model=List[SensorDataResponse])
    async def get_sensor_data(limit: int = 100, offset: int = 0):
        """Get paginated sensor data"""
        try:
            db_manager = DatabaseManager()
            data = db_manager.get_sensor_data(limit=limit, offset=offset)
            return [SensorDataResponse(**item) for item in data]
        except Exception as e:
            logger.error(f"❌ Database error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    @app.get("/api/alerts", response_model=List[AlertResponse])
    async def get_alerts(limit: int = 100, offset: int = 0, status: Optional[str] = None):
        """Get paginated alerts"""
        try:
            db_manager = DatabaseManager()
            alerts = db_manager.get_alerts(limit=limit, offset=offset, status=status)
            return [AlertResponse(**alert) for alert in alerts]
        except Exception as e:
            logger.error(f"❌ Database error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    @app.get("/api/predictions", response_model=List[PredictionResponse])
    async def get_predictions(limit: int = 100, offset: int = 0):
        """Get paginated predictions"""
        try:
            db_manager = DatabaseManager()
            predictions = db_manager.get_predictions(limit=limit, offset=offset)
            return [PredictionResponse(**pred) for pred in predictions]
        except Exception as e:
            logger.error(f"❌ Database error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    @app.get("/api/equipment/{equipment_id}/stats", response_model=EquipmentStatsResponse)
    async def get_equipment_stats(equipment_id: str):
        """Get statistics for specific equipment"""
        try:
            db_manager = DatabaseManager()
            stats = db_manager.get_equipment_stats(equipment_id)
            return EquipmentStatsResponse(**stats)
        except Exception as e:
            logger.error(f"❌ Database error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    @app.post("/api/alerts/{alert_id}/resolve")
    async def resolve_alert(alert_id: int):
        """Resolve an alert"""
        try:
            db_manager = DatabaseManager()
            success = db_manager.resolve_alert(alert_id)
            if success:
                return {"message": f"Alert {alert_id} resolved", "status": "success"}
            else:
                raise HTTPException(status_code=404, detail="Alert not found")
        except Exception as e:
            logger.error(f"❌ Database error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    @app.delete("/api/cleanup")
    async def cleanup_old_data():
        """Clean up old data from database"""
        try:
            db_manager = DatabaseManager()
            counts = db_manager.cleanup_old_data()
            return {
                "message": "Cleanup completed",
                "removed_records": counts,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Cleanup error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")
    
    @app.get("/api/database/health")
    async def get_database_health():
        """Get database health and statistics"""
        try:
            db_manager = DatabaseManager()
            health = db_manager.health_check()
            return health
        except Exception as e:
            logger.error(f"❌ Database health check error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# ===================================================================
# 6. TESTING ENDPOINTS (Simulation, Testing)
# ===================================================================

def setup_testing_endpoints(app, ml_detector, alert_manager):
    """Setup testing and simulation endpoints"""
    
    @app.post("/simulate_data")
    async def simulate_sensor_data():
        """Generate simulated sensor data for testing"""
        try:
            import random
            
            # Random equipment types for simulation
            equipment_types = ["POMPE", "SOUPAPE", "VENTILATEUR", "CONDENSEUR", "VANNE", "TURBINE", "GENERATEUR"]
            equipment_type = random.choice(equipment_types)
            
            # Generate realistic sensor data based on equipment type
            base_profiles = ml_detector.equipment_profiles.get(equipment_type, ml_detector.equipment_profiles["POMPE"])
            
            # Add some variation to create interesting data
            anomaly_factor = random.uniform(0.8, 2.5)  # Some readings will be anomalous
            
            sensor_data = SensorData(
                equipment_id=f"SIM_{equipment_type}_{random.randint(1000, 9999)}",
                equipment_type=equipment_type,
                temperature=base_profiles["normal_ranges"]["temperature"]["min"] + 
                           random.uniform(0, base_profiles["normal_ranges"]["temperature"]["max"] - 
                                        base_profiles["normal_ranges"]["temperature"]["min"]) * anomaly_factor,
                pressure=base_profiles["normal_ranges"]["pressure"]["min"] + 
                        random.uniform(0, base_profiles["normal_ranges"]["pressure"]["max"] - 
                                     base_profiles["normal_ranges"]["pressure"]["min"]) * anomaly_factor,
                vibration=random.uniform(0.1, 15.0) * anomaly_factor,
                flow_rate=random.uniform(50, 200) * anomaly_factor,
                power_consumption=random.uniform(100, 1000) * anomaly_factor,
                operating_hours=random.uniform(0, 24),
                location="SIMULATION",
                timestamp=datetime.now().isoformat()
            )
            
            # Get ML prediction for the simulated data
            prediction = ml_enhanced_anomaly_detection(sensor_data, ml_detector, alert_manager, logger)
            
            return {
                "message": "🎲 Simulated sensor data generated",
                "sensor_data": sensor_data.model_dump(),
                "ml_prediction": prediction.model_dump(),
                "simulation_info": {
                    "anomaly_factor_applied": anomaly_factor,
                    "expected_anomaly": anomaly_factor > 1.5,
                    "equipment_profile_used": equipment_type
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Simulation error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
    
    @app.get("/test")
    async def test_all_systems():
        """Comprehensive system test"""
        try:
            test_results = {
                "timestamp": datetime.now().isoformat(),
                "ml_detector": {
                    "status": "trained" if ml_detector.is_trained else "not_trained",
                    "algorithms_available": list(ml_detector.algorithms.keys()),
                    "equipment_profiles": len(ml_detector.equipment_profiles)
                },
                "alert_manager": {
                    "status": "active",
                    "active_alerts_count": len(alert_manager.active_alerts)
                },
                "database": {},
                "overall_status": "healthy"
            }
            
            # Test database connection
            try:
                db_manager = DatabaseManager()
                test_results["database"] = db_manager.health_check()
            except Exception as db_error:
                test_results["database"] = {"status": "error", "error": str(db_error)}
                test_results["overall_status"] = "partial"
            
            # Test ML prediction with sample data
            try:
                sample_data = SensorData(
                    equipment_id="TEST_001",
                    equipment_type="POMPE",
                    temperature=75.0,
                    pressure=150.0,
                    vibration=2.5,
                    flow_rate=120.0,
                    power_consumption=500.0,
                    operating_hours=8.0,
                    location="TEST_LAB",
                    timestamp=datetime.now().isoformat()
                )
                
                prediction = ml_enhanced_anomaly_detection(sample_data, ml_detector, alert_manager, logger)
                test_results["ml_prediction_test"] = {
                    "status": "success",
                    "prediction_generated": True,
                    "anomaly_detected": prediction.is_anomaly,
                    "confidence": prediction.confidence
                }
                
            except Exception as ml_error:
                test_results["ml_prediction_test"] = {"status": "error", "error": str(ml_error)}
                test_results["overall_status"] = "partial"
            
            return test_results
            
        except Exception as e:
            logger.error(f"❌ System test error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"System test failed: {str(e)}")

# ===================================================================
# 7. MAIN ROUTE SETUP FUNCTION
# ===================================================================

def setup_api_routes(app, ml_detector, alert_manager):
    """
    Setup all API routes for the TAQA ML-Enhanced Anomaly Detection System
    
    Args:
        app: FastAPI application instance
        ml_detector: TAQAMLAnomalyDetector instance
        alert_manager: AlertManager instance
    """
    logger.info("🔧 Setting up API routes...")
    
    # Setup all endpoint groups
    setup_core_endpoints(app, ml_detector, alert_manager)
    setup_data_processing_endpoints(app, ml_detector, alert_manager)
    setup_monitoring_endpoints(app, ml_detector, alert_manager)
    setup_database_endpoints(app, ml_detector, alert_manager)
    setup_testing_endpoints(app, ml_detector, alert_manager)
    
    # Setup startup event
    @app.on_event("startup")
    async def startup_event():
        """Initialize system on startup"""
        logger.info("🚀 TAQA ML-Enhanced Anomaly Detection System Starting...")
        logger.info("🔧 Initializing components...")
        
        logger.info("🚨 Alert Manager initialized")
        logger.info("🤖 ML Detector initialized")
        logger.info(f"🔧 {len(ml_detector.equipment_profiles)} equipment profiles loaded")
        
        # Try to load existing models
        if ml_detector.load_models():
            logger.info("✅ Pre-trained ML models loaded successfully!")
            logger.info("🧠 Real ML algorithms ready for predictions")
        else:
            logger.info("⚠️ No pre-trained models found")
            logger.info("🔄 Use /train_ml endpoint to train ML models")
            logger.info("📋 Fallback to rule-based detection available")
        
        logger.info("🔧 ALL ISSUES FIXED: Serialization, NumPy deprecation, ML training")
        logger.info("📖 API Documentation: http://localhost:8000/docs")
    
    logger.info("✅ All API routes configured successfully!")
    logger.info("📡 Available endpoint groups:")
    logger.info("   • Core endpoints: /, /predict, /train_ml, /ml_status")
    logger.info("   • Data processing: /upload_excel")
    logger.info("   • Monitoring: /health, /model_stats, /alerts/active")
    logger.info("   • Database API: /api/sensor-data, /api/alerts, /api/predictions")
    logger.info("   • Testing: /simulate_data, /test")
