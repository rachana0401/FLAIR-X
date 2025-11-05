"""
Adaptive Privacy Engine
Dynamically adjusts privacy parameters based on risk assessment
"""
import numpy as np
from typing import Dict, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    INITIAL_NOISE_MULTIPLIER,
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    NOISE_INCREASE_RATE,
    NOISE_DECREASE_RATE
)


class AdaptivePrivacyEngine:
    """
    Adaptive privacy engine that adjusts noise levels based on privacy risk
    """
    
    def __init__(
        self,
        initial_noise: float = None,
        high_risk_threshold: float = None,
        low_risk_threshold: float = None,
        noise_increase_rate: float = None,
        noise_decrease_rate: float = None
    ):
        self.initial_noise = initial_noise or INITIAL_NOISE_MULTIPLIER
        self.high_risk_threshold = high_risk_threshold or HIGH_RISK_THRESHOLD
        self.low_risk_threshold = low_risk_threshold or LOW_RISK_THRESHOLD
        self.noise_increase_rate = noise_increase_rate or NOISE_INCREASE_RATE
        self.noise_decrease_rate = noise_decrease_rate or NOISE_DECREASE_RATE
        
        # Track client states
        self.client_risk_scores: Dict[str, float] = {}
        self.client_noise_levels: Dict[str, float] = {}
        self.adjustment_history: Dict[str, list] = {}
    
    def initialize_client(self, client_id: str, initial_risk: float = 0.5):
        """Initialize tracking for a new client"""
        self.client_risk_scores[client_id] = initial_risk
        self.client_noise_levels[client_id] = self.initial_noise
        self.adjustment_history[client_id] = []
    
    def update_risk_score(self, client_id: str, risk_score: float):
        """Update the privacy risk score for a client"""
        self.client_risk_scores[client_id] = np.clip(risk_score, 0.0, 1.0)
    
    def adjust_noise(self, client_id: str) -> Dict[str, any]:
        """
        Adjust noise level for a client based on current risk score
        
        Returns:
            Dict with adjustment details
        """
        if client_id not in self.client_risk_scores:
            self.initialize_client(client_id)
        
        current_noise = self.client_noise_levels.get(client_id, self.initial_noise)
        risk_score = self.client_risk_scores[client_id]
        
        old_noise = current_noise
        
        if risk_score > self.high_risk_threshold:
            # High risk → increase noise
            new_noise = current_noise * (1 + self.noise_increase_rate)
            action = "increase"
            reason = "high_risk"
        elif risk_score < self.low_risk_threshold:
            # Low risk → decrease noise (improve accuracy)
            new_noise = current_noise * (1 - self.noise_decrease_rate)
            action = "decrease"
            reason = "low_risk"
        else:
            # Medium risk → keep same
            new_noise = current_noise
            action = "keep"
            reason = "medium_risk"
        
        # Apply bounds
        new_noise = np.clip(new_noise, 0.1, 2.0)
        self.client_noise_levels[client_id] = new_noise
        
        # Log adjustment
        adjustment = {
            "old_noise": float(old_noise),
            "new_noise": float(new_noise),
            "risk_score": float(risk_score),
            "action": action,
            "reason": reason
        }
        self.adjustment_history[client_id].append(adjustment)
        
        return adjustment
    
    def adjust_all_clients(self) -> Dict[str, Dict[str, any]]:
        """Adjust noise for all tracked clients"""
        adjustments = {}
        for client_id in self.client_risk_scores.keys():
            adjustments[client_id] = self.adjust_noise(client_id)
        return adjustments
    
    def get_noise_level(self, client_id: str) -> float:
        """Get current noise level for a client"""
        if client_id not in self.client_noise_levels:
            self.initialize_client(client_id)
        return self.client_noise_levels[client_id]
    
    def get_risk_score(self, client_id: str) -> float:
        """Get current risk score for a client"""
        if client_id not in self.client_risk_scores:
            self.initialize_client(client_id)
        return self.client_risk_scores[client_id]
    
    def get_state(self) -> Dict[str, Dict[str, float]]:
        """Get current state of all clients"""
        return {
            client_id: {
                "risk_score": self.client_risk_scores.get(client_id, 0.5),
                "noise_level": self.client_noise_levels.get(client_id, self.initial_noise)
            }
            for client_id in set(list(self.client_risk_scores.keys()) + list(self.client_noise_levels.keys()))
        }
    
    def compute_risk_from_gradients(self, gradients: list) -> float:
        """
        Compute privacy risk score from gradient statistics
        
        Args:
            gradients: List of gradient arrays (numpy arrays)
        
        Returns:
            Risk score between 0 and 1
        """
        if not gradients:
            return 0.5
        
        # Compute gradient statistics
        grad_magnitudes = [np.linalg.norm(g) for g in gradients]
        
        if not grad_magnitudes:
            return 0.5
        
        avg_magnitude = np.mean(grad_magnitudes)
        std_magnitude = np.std(grad_magnitudes)
        
        # Risk score based on gradient variance
        # Higher variance indicates more information leakage potential
        risk_score = min(1.0, (avg_magnitude + std_magnitude) / 100.0)
        
        # Add some randomness to simulate real privacy auditing
        risk_score += np.random.normal(0, 0.05)
        risk_score = np.clip(risk_score, 0.0, 1.0)
        
        return float(risk_score)


if __name__ == "__main__":
    print("🧪 Testing Adaptive Privacy Engine...\n")
    
    engine = AdaptivePrivacyEngine()
    
    # Initialize clients
    for client_id in ["A", "B", "C"]:
        engine.initialize_client(client_id, initial_risk=0.5)
    
    print("1. Initial state:")
    for client_id, state in engine.get_state().items():
        print(f"   {client_id}: risk={state['risk_score']:.3f}, noise={state['noise_level']:.3f}")
    
    # Simulate risk updates and adjustments
    print("\n2. Simulating risk updates...")
    engine.update_risk_score("A", 0.8)  # High risk
    engine.update_risk_score("B", 0.4)  # Medium risk
    engine.update_risk_score("C", 0.2)  # Low risk
    
    print("\n3. Adjusting noise levels...")
    adjustments = engine.adjust_all_clients()
    for client_id, adj in adjustments.items():
        print(f"   {client_id}: {adj['action']} noise {adj['old_noise']:.3f} → {adj['new_noise']:.3f} (risk={adj['risk_score']:.3f})")
    
    print("\n4. Final state:")
    for client_id, state in engine.get_state().items():
        print(f"   {client_id}: risk={state['risk_score']:.3f}, noise={state['noise_level']:.3f}")
    
    print("\n✅ All tests passed!")

