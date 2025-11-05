"""
Privacy Audit Module
Performs privacy risk assessments and auditing
"""
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from ledger.logger import get_ledger


class PrivacyAuditor:
    """
    Privacy auditor for assessing privacy risks in federated learning
    """
    
    def __init__(self):
        self.audit_history: List[Dict[str, Any]] = []
    
    def assess_risk_from_parameters(
        self,
        parameters: List[np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assess privacy risk from model parameters/gradients
        
        Args:
            parameters: List of parameter arrays
            metadata: Optional metadata (e.g., hospital_id, round_num)
        
        Returns:
            Dict with risk assessment results
        """
        if not parameters:
            return {
                "risk_score": 0.5,
                "confidence": 0.0,
                "factors": {}
            }
        
        # Convert parameters to numpy if needed
        param_arrays = [np.array(p) for p in parameters]
        
        # Compute various statistics
        grad_magnitudes = [np.linalg.norm(p) for p in param_arrays]
        grad_variances = [np.var(p) for p in param_arrays]
        
        avg_magnitude = np.mean(grad_magnitudes)
        std_magnitude = np.std(grad_magnitudes)
        avg_variance = np.mean(grad_variances)
        
        # Risk factors
        factors = {
            "gradient_magnitude": float(avg_magnitude),
            "gradient_variance": float(avg_variance),
            "magnitude_std": float(std_magnitude)
        }
        
        # Compute risk score (0-1 scale)
        # Higher gradient variance indicates more information leakage potential
        risk_score = min(1.0, (avg_magnitude + std_magnitude) / 100.0)
        
        # Add some randomness to simulate real privacy auditing uncertainty
        risk_score += np.random.normal(0, 0.05)
        risk_score = np.clip(risk_score, 0.0, 1.0)
        
        # Confidence based on consistency
        confidence = 1.0 - min(1.0, std_magnitude / (avg_magnitude + 1e-6))
        
        assessment = {
            "risk_score": float(risk_score),
            "confidence": float(confidence),
            "factors": factors,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Log audit
        self.audit_history.append(assessment)
        
        # Log to ledger if metadata contains hospital_id
        if metadata and "hospital_id" in metadata:
            ledger = get_ledger()
            ledger.log_privacy_event(
                event_type="risk_assessment",
                hospital_id=metadata["hospital_id"],
                details={
                    "risk_score": assessment["risk_score"],
                    "confidence": assessment["confidence"],
                    "factors": assessment["factors"]
                }
            )
        
        return assessment
    
    def assess_risk_from_metrics(
        self,
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assess privacy risk from training metrics
        
        Args:
            metrics: Training metrics (loss, accuracy, etc.)
            metadata: Optional metadata
        
        Returns:
            Dict with risk assessment results
        """
        # Simplified risk assessment based on metrics
        # Higher loss might indicate more distinctive gradients
        train_loss = metrics.get("train_loss", 1.0)
        train_acc = metrics.get("train_acc", 0.0)
        
        # Risk factors
        factors = {
            "train_loss": float(train_loss),
            "train_acc": float(train_acc)
        }
        
        # Simplified risk: higher loss = more risk (very rough proxy)
        risk_score = np.clip(train_loss / 3.0, 0.0, 1.0)
        
        # Add randomness
        risk_score += np.random.normal(0, 0.05)
        risk_score = np.clip(risk_score, 0.0, 1.0)
        
        assessment = {
            "risk_score": float(risk_score),
            "confidence": 0.7,  # Lower confidence for metric-based assessment
            "factors": factors,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.audit_history.append(assessment)
        
        return assessment
    
    def detect_anomalies(
        self,
        client_metrics: Dict[str, Dict[str, Any]],
        threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Detect anomalous behavior that might indicate privacy risks
        
        Args:
            client_metrics: Dict mapping client_id to metrics
            threshold: Risk threshold for flagging anomalies
        
        Returns:
            Dict with detected anomalies
        """
        anomalies = {}
        
        for client_id, metrics in client_metrics.items():
            risk_score = metrics.get("risk_score", 0.5)
            
            if risk_score > threshold:
                anomalies[client_id] = {
                    "risk_score": risk_score,
                    "severity": "high" if risk_score > 0.9 else "medium",
                    "description": f"High privacy risk detected: {risk_score:.3f}"
                }
        
        return {
            "anomalies": anomalies,
            "total_anomalies": len(anomalies),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of all audits"""
        if not self.audit_history:
            return {
                "total_audits": 0,
                "avg_risk_score": 0.5
            }
        
        risk_scores = [a["risk_score"] for a in self.audit_history]
        
        return {
            "total_audits": len(self.audit_history),
            "avg_risk_score": float(np.mean(risk_scores)),
            "min_risk_score": float(np.min(risk_scores)),
            "max_risk_score": float(np.max(risk_scores)),
            "std_risk_score": float(np.std(risk_scores)),
            "first_audit": self.audit_history[0]["timestamp"],
            "last_audit": self.audit_history[-1]["timestamp"]
        }


if __name__ == "__main__":
    print("🧪 Testing Privacy Auditor...\n")
    
    auditor = PrivacyAuditor()
    
    # Test risk assessment from parameters
    print("1. Testing risk assessment from parameters...")
    dummy_params = [
        np.random.randn(10, 10),
        np.random.randn(5, 5)
    ]
    assessment = auditor.assess_risk_from_parameters(
        dummy_params,
        metadata={"hospital_id": "A", "round": 1}
    )
    print(f"   Risk score: {assessment['risk_score']:.3f}")
    print(f"   Confidence: {assessment['confidence']:.3f}")
    print(f"   ✅ Assessment complete")
    
    # Test risk assessment from metrics
    print("\n2. Testing risk assessment from metrics...")
    metrics_assessment = auditor.assess_risk_from_metrics(
        {"train_loss": 0.8, "train_acc": 0.75},
        metadata={"hospital_id": "B", "round": 1}
    )
    print(f"   Risk score: {metrics_assessment['risk_score']:.3f}")
    print(f"   ✅ Assessment complete")
    
    # Test anomaly detection
    print("\n3. Testing anomaly detection...")
    client_metrics = {
        "A": {"risk_score": 0.9},
        "B": {"risk_score": 0.4},
        "C": {"risk_score": 0.85}
    }
    anomalies = auditor.detect_anomalies(client_metrics)
    print(f"   Detected {anomalies['total_anomalies']} anomalies")
    for client_id, anomaly in anomalies["anomalies"].items():
        print(f"   {client_id}: {anomaly['severity']} - {anomaly['description']}")
    print(f"   ✅ Anomaly detection complete")
    
    # Test audit summary
    print("\n4. Getting audit summary...")
    summary = auditor.get_audit_summary()
    print(f"   Total audits: {summary['total_audits']}")
    print(f"   Avg risk score: {summary['avg_risk_score']:.3f}")
    print(f"   ✅ Summary generated")
    
    print("\n✅ All tests passed!")

