# config.py - Configuration and Constants
"""
Configuration file for TAQA ML-Enhanced Anomaly Detection System
Contains all system settings, equipment profiles, and constants
"""

from typing import Dict, List, Tuple
import logging

# System Configuration
APP_CONFIG = {
    "API_TITLE": "TAQA ML-Enhanced Anomaly Detection API",
    "VERSION": "4.0.2",
    "DESCRIPTION": "Real ML algorithms: Random Forest, SVM, Neural Networks, Isolation Forest, etc.",
    "HOST": "0.0.0.0",
    "PORT": 8000,
    "DEBUG": True,
    
    # ML Settings
    "ML_TRAINING_SAMPLES": 5000,
    "ML_TRAINING_TIMEOUT": 300,  # 5 minutes
    "ANOMALY_THRESHOLD": 70.0,  # Score above which alerts are triggered
    
    # File Processing
    "MAX_FILE_SIZE": 50 * 1024 * 1024,  # 50MB
    "SUPPORTED_FORMATS": ['.xlsx', '.xls', '.csv'],
    "BATCH_PROCESSING_SIZE": 500  # Progress update interval
}

# Equipment Profiles - Industrial Knowledge Base
EQUIPMENT_PROFILES = {
    "POMPE": {
        "normal_ranges": {
            "temperature": (35, 85),    # °C
            "pressure": (8, 30),        # bar
            "vibration": (1.0, 4.5),    # mm/s
            "efficiency": (85, 98)       # %
        },
        "critical_thresholds": {
            "temperature": 100,
            "pressure": 35,
            "vibration": 6.0,
            "efficiency": 70
        },
        "failure_modes": ["cavitation", "bearing_wear", "seal_leak", "overheating", "motor_fault"],
        "maintenance_costs": {"preventive": 2000, "corrective": 8000, "emergency": 25000}
    },
    "SOUPAPE": {
        "normal_ranges": {
            "temperature": (20, 75),
            "pressure": (5, 25),
            "vibration": (0.3, 3.0),
            "efficiency": (80, 95)
        },
        "critical_thresholds": {
            "temperature": 90,
            "pressure": 30,
            "vibration": 4.0,
            "efficiency": 65
        },
        "failure_modes": ["stuck_open", "stuck_closed", "leakage", "actuator_fault"],
        "maintenance_costs": {"preventive": 1500, "corrective": 6000, "emergency": 20000}
    },
    "VENTILATEUR": {
        "normal_ranges": {
            "temperature": (30, 80),
            "pressure": (3, 20),
            "vibration": (1.5, 5.0),
            "efficiency": (78, 92)
        },
        "critical_thresholds": {
            "temperature": 95,
            "pressure": 25,
            "vibration": 7.0,
            "efficiency": 60
        },
        "failure_modes": ["blade_damage", "motor_failure", "bearing_wear", "imbalance"],
        "maintenance_costs": {"preventive": 2500, "corrective": 10000, "emergency": 30000}
    },
    "CONDENSEUR": {
        "normal_ranges": {
            "temperature": (25, 70),
            "pressure": (10, 28),
            "vibration": (0.5, 2.5),
            "efficiency": (88, 98)
        },
        "critical_thresholds": {
            "temperature": 85,
            "pressure": 35,
            "vibration": 3.5,
            "efficiency": 75
        },
        "failure_modes": ["tube_fouling", "tube_leak", "corrosion", "scaling"],
        "maintenance_costs": {"preventive": 3000, "corrective": 12000, "emergency": 35000}
    },
    "VANNE": {
        "normal_ranges": {
            "temperature": (15, 65),
            "pressure": (4, 22),
            "vibration": (0.2, 2.0),
            "efficiency": (75, 95)
        },
        "critical_thresholds": {
            "temperature": 80,
            "pressure": 28,
            "vibration": 3.0,
            "efficiency": 60
        },
        "failure_modes": ["actuator_failure", "seat_leakage", "stem_binding"],
        "maintenance_costs": {"preventive": 1800, "corrective": 7000, "emergency": 22000}
    },
    "TURBINE": {
        "normal_ranges": {
            "temperature": (40, 90),
            "pressure": (15, 40),
            "vibration": (2.0, 6.0),
            "efficiency": (90, 98)
        },
        "critical_thresholds": {
            "temperature": 110,
            "pressure": 50,
            "vibration": 8.0,
            "efficiency": 80
        },
        "failure_modes": ["blade_erosion", "bearing_failure", "seal_wear"],
        "maintenance_costs": {"preventive": 5000, "corrective": 20000, "emergency": 60000}
    },
    "GENERATEUR": {
        "normal_ranges": {
            "temperature": (45, 95),
            "pressure": (0, 5),
            "vibration": (1.0, 4.0),
            "efficiency": (92, 99)
        },
        "critical_thresholds": {
            "temperature": 115,
            "pressure": 8,
            "vibration": 6.0,
            "efficiency": 85
        },
        "failure_modes": ["winding_fault", "bearing_failure", "cooling_failure"],
        "maintenance_costs": {"preventive": 5000, "corrective": 20000, "emergency": 60000}
    }
}

# Alert Cost Configuration
SEVERITY_COSTS = {
    "CRITICAL": 50000,
    "HIGH": 20000,
    "MEDIUM": 5000,
    "LOW": 1000
}

# Equipment Type Keywords for Text Analysis
EQUIPMENT_KEYWORDS = {
    "POMPE": ["pompe", "pump"],
    "SOUPAPE": ["soupape", "valve", "clapet"],
    "VENTILATEUR": ["ventilateur", "fan", "soufflante", "tirage"],
    "CONDENSEUR": ["condenseur", "condenser"],
    "TURBINE": ["turbine"],
    "GENERATEUR": ["générateur", "generator", "alternateur"],
    "VANNE": ["vanne", "valve"]
}

# Failure Pattern Keywords
FAILURE_KEYWORDS = {
    "overheating": ["surchauffe", "température", "chaud", "overheating", "hot"],
    "pressure_issues": ["fuite", "étanche", "pression", "leak", "pressure"],
    "vibration_issues": ["vibration", "palier", "bearing", "bruit", "noise"],
    "performance_issues": ["colmatage", "encrassement", "performance", "rendement"],
    "critical_failure": ["défaillance", "panne", "arrêt", "failure", "breakdown"]
}

# ML Model Configuration
ML_CONFIG = {
    "model_weights": {
        "random_forest": 0.25,
        "svm_classifier": 0.20,
        "neural_network": 0.20,
        "isolation_forest": 0.15,
        "one_class_svm": 0.10,
        "local_outlier_factor": 0.10
    },
    "feature_names": [
        "temperature", "pressure", "vibration", "efficiency",
        "load_factor", "hour", "day_of_week", "month", "season",
        "ambient_temperature", "humidity", "equipment_age",
        "temp_pressure_ratio", "vibration_efficiency_ratio",
        "power_factor", "thermal_stress", "mechanical_stress",
        "equipment_type_encoded"
    ],
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    },
    "svm": {
        "kernel": "rbf",
        "gamma": "scale",
        "C": 1.0
    },
    "neural_network": {
        "hidden_layer_sizes": (100, 50, 25),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "max_iter": 500
    },
    "isolation_forest": {
        "contamination": 0.30,
        "n_estimators": 200
    },
    "autoencoder": {
        "hidden_layer_sizes": (50, 20, 10, 20, 50)
    }
}

# Logging Configuration
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)
