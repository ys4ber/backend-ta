# main.py - TAQA ML-Enhanced Anomaly Detection API
# Clean FastAPI application with separated concerns

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Import configuration and core components
from config import APP_CONFIG
from alert_manager import AlertManager
from ml_models import TAQAMLAnomalyDetector
from api import setup_api_routes

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🚀 Starting TAQA ML-Enhanced Anomaly Detection System...")

# Initialize FastAPI app
app = FastAPI(
    title=APP_CONFIG["API_TITLE"], 
    version=APP_CONFIG["VERSION"],
    description=APP_CONFIG["DESCRIPTION"]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global instances
ml_detector = TAQAMLAnomalyDetector()
alert_manager = AlertManager()

# Setup API routes
setup_api_routes(app, ml_detector, alert_manager)

if __name__ == "__main__":
    print("🏭 Starting TAQA ML-Enhanced Anomaly Detection Server...")
    print("🤖 Features: REAL ML ALGORITHMS + Enhanced Alerts + Predictive Analytics")
    print("🌐 Server: http://localhost:8000")
    print("🧠 ML Algorithms: Random Forest, SVM, Neural Network, Isolation Forest, One-Class SVM, LOF, Autoencoder")
    print("🔔 Alert System: ML-enhanced with confidence scoring")
    print("📊 Dependencies: scikit-learn, pandas, numpy, fastapi")
    print("🔧 Equipment Types: POMPE, SOUPAPE, VENTILATEUR, CONDENSEUR, VANNE, TURBINE, GENERATEUR")
    print("✅ Ready for real ML-based anomaly detection!")
    print("🎯 COMPLETELY FIXED: All serialization and training issues resolved!")
    print("📁 API Structure: All endpoints organized in api.py")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )