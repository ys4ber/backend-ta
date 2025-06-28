# alert_manager.py - Alert Management System
"""
Alert management system for equipment monitoring
Handles alert creation, tracking, and cost estimation
"""

from typing import List, Dict
from datetime import datetime
import logging
from models import Alert
from config import SEVERITY_COSTS
from utils import safe_convert_numpy

logger = logging.getLogger(__name__)

class AlertManager:
    """
    Manages equipment alerts and notifications
    Tracks active alerts and maintains alert history
    """
    
    def __init__(self):
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        
    def create_alert(self, equipment_id: str, equipment_type: str, severity: str, 
                    message: str, sensor_data: Dict, recommendations: List[str]) -> Alert:
        """
        Create a new alert for equipment
        
        Args:
            equipment_id: Equipment identifier
            equipment_type: Type of equipment
            severity: Alert severity (CRITICAL, HIGH, MEDIUM, LOW)
            message: Alert message
            sensor_data: Sensor readings that triggered alert
            recommendations: Recommended actions
            
        Returns:
            Created Alert object
        """
        # Generate unique alert ID
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{equipment_id}"
        
        # Get estimated cost based on severity
        estimated_cost = SEVERITY_COSTS.get(severity, 5000)
        
        # Create alert object
        alert = Alert(
            id=alert_id,
            equipment_id=equipment_id,
            equipment_type=equipment_type,
            severity=severity,
            title=f"{severity} Alert - {equipment_id}",
            message=message,
            triggered_at=datetime.now().isoformat(),
            sensor_data=sensor_data,
            recommended_actions=recommendations,
            estimated_cost=estimated_cost
        )
        
        # Add to active alerts and history
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Log alert creation
        logger.info(f"🚨 ALERT: {severity} - {equipment_id}")
        
        return alert
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts"""
        return self.active_alerts.copy()
    
    def get_alert_history(self, limit: int = None) -> List[Alert]:
        """
        Get alert history
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of historical alerts
        """
        if limit:
            return self.alert_history[-limit:]
        return self.alert_history.copy()
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Mark an alert as resolved (remove from active alerts)
        
        Args:
            alert_id: ID of alert to resolve
            
        Returns:
            True if alert was found and resolved
        """
        for i, alert in enumerate(self.active_alerts):
            if alert.id == alert_id:
                resolved_alert = self.active_alerts.pop(i)
                logger.info(f"✅ RESOLVED: Alert {alert_id} for {resolved_alert.equipment_id}")
                return True
        return False
    
    def get_alerts_by_equipment(self, equipment_id: str) -> List[Alert]:
        """
        Get all alerts for specific equipment
        
        Args:
            equipment_id: Equipment identifier
            
        Returns:
            List of alerts for the equipment
        """
        return [alert for alert in self.alert_history if alert.equipment_id == equipment_id]
    
    def get_alerts_by_severity(self, severity: str) -> List[Alert]:
        """
        Get all active alerts by severity level
        
        Args:
            severity: Severity level to filter by
            
        Returns:
            List of alerts with specified severity
        """
        return [alert for alert in self.active_alerts if alert.severity == severity]
    
    def get_alert_statistics(self) -> Dict:
        """
        Get alert statistics
        
        Returns:
            Dictionary with alert statistics
        """
        severity_counts = {}
        for alert in self.active_alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        total_estimated_cost = sum(alert.estimated_cost or 0 for alert in self.active_alerts)
        
        return {
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "severity_breakdown": severity_counts,
            "total_estimated_cost": total_estimated_cost,
            "average_cost_per_alert": total_estimated_cost / max(1, len(self.active_alerts))
        }
    
    def clear_all_alerts(self):
        """Clear all active alerts (for testing purposes)"""
        self.active_alerts.clear()
        logger.info("🧹 All active alerts cleared")
    
    def export_alerts_data(self) -> Dict:
        """
        Export alerts data in serializable format
        
        Returns:
            Serializable dictionary of alerts data
        """
        return {
            "active_alerts": [safe_convert_numpy(alert.model_dump()) for alert in self.active_alerts],
            "active_count": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "recent_alerts": [safe_convert_numpy(alert.model_dump()) for alert in self.alert_history[-10:]],
            "statistics": self.get_alert_statistics()
        }

# Global alert manager instance
alert_manager = AlertManager()
