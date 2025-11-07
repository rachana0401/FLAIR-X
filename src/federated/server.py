"""
Custom Flower Server Strategy with Adaptive Privacy
Dynamically adjusts noise levels based on privacy risk
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, Scalar
from typing import List, Tuple, Dict, Optional, Union
import numpy as np

from config import (
    INITIAL_NOISE_MULTIPLIER,
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    NOISE_INCREASE_RATE,
    NOISE_DECREASE_RATE
)
from ledger.logger import get_ledger


class AdaptivePrivacyStrategy(FedAvg):
    """
    Custom FedAvg strategy with adaptive differential privacy
    Adjusts noise levels per client based on privacy risk scores
    """
    
    def __init__(self):
        super().__init__(
            fraction_fit=1.0,  # Use all available clients
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
        )
        
        # Track privacy metrics per client
        self.client_risk_scores = {}  # hospital_id -> risk_score
        self.client_noise_levels = {}  # hospital_id -> noise_multiplier
        
        # Initialize noise levels
        for hospital_id in ["A", "B", "C"]:
            self.client_noise_levels[hospital_id] = INITIAL_NOISE_MULTIPLIER
            self.client_risk_scores[hospital_id] = 0.5  # Start with medium risk
        
        # Track metrics over rounds
        self.round_metrics = []
        
        print("🌐 Adaptive Privacy Strategy initialized")
        print(f"   Initial noise: {INITIAL_NOISE_MULTIPLIER}")
    
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager
    ):
        """
        Configure the next round of training
        Send different noise levels to each client based on risk
        """
        config = {}
        
        # Get available clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients
        )
        
        # Create client configs with adaptive noise
        fit_configs = []
        for client in clients:
            # Client ID is stored in client.cid (e.g., "0", "1", "2")
            # Map to hospital IDs
            hospital_id = chr(65 + int(client.cid))  # 0->A, 1->B, 2->C
            
            # Get current noise level for this hospital
            noise_level = self.client_noise_levels.get(hospital_id, INITIAL_NOISE_MULTIPLIER)
            
            client_config = {
                "noise_multiplier": noise_level,
                "server_round": server_round
            }
            
            fit_configs.append((client, client_config))
        
        print(f"\n🔄 Round {server_round} - Configured clients:")
        for client, cfg in fit_configs:
            hospital_id = chr(65 + int(client.cid))
            risk = self.client_risk_scores.get(hospital_id, 0.5)
            print(f"   Hospital {hospital_id}: noise={cfg['noise_multiplier']:.3f}, risk={risk:.3f}")
        
        return fit_configs
    
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures
    ):
        """
        Aggregate client updates and compute privacy risks
        """
        print(f"\n📊 Round {server_round} - Aggregating results...")
        
        if failures:
            print(f"   ⚠️ {len(failures)} clients failed")
        
        # Compute privacy risk scores from gradients
        for client, fit_res in results:
            hospital_id = chr(65 + int(client.cid))
            
            # Extract metrics
            metrics = fit_res.metrics
            
            # Compute risk score (simplified - based on gradient magnitude)
            risk_score = self._compute_risk_score(fit_res.parameters, metrics)
            self.client_risk_scores[hospital_id] = risk_score
            
            print(f"   Hospital {hospital_id}: risk={risk_score:.3f}, "
                  f"acc={metrics.get('train_acc', 0):.3f}")
        
        # Adjust noise levels for next round
        self._adjust_noise_levels()
        
        # Perform standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Log round metrics
        self._log_round_metrics(server_round, results)
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures
    ):
        """
        Aggregate evaluation results
        """
        if failures:
            print(f"   ⚠️ Evaluation failures: {len(failures)}")
        
        # Standard aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Print evaluation results
        if results:
            avg_acc = np.mean([r[1].metrics.get("val_acc", 0) for r in results])
            print(f"   Global Val Accuracy: {avg_acc:.4f}")
        
        return aggregated_loss, aggregated_metrics
    
    from intelligent_risk import IntelligentRiskCalculator, SmoothNoiseAdjuster

class AdaptivePrivacyStrategy(FedAvg):
    def __init__(self):
        super().__init__(...)
        
        # Replace simple risk tracking with intelligent calculator
        self.risk_calculator = IntelligentRiskCalculator()
        self.noise_adjuster = SmoothNoiseAdjuster()
    
    def aggregate_fit(self, server_round, results, failures):
        for client, fit_res in results:
            hospital_id = chr(65 + int(client.cid))
            
            # Use intelligent risk calculator
            risk_result = self.risk_calculator.compute_risk(
                hospital_id,
                fit_res.parameters.tensors,
                server_round
            )
            
            # Get trend
            trend = self.risk_calculator.get_risk_trend(hospital_id)
            
            # Smooth noise adjustment
            noise_adjustment = self.noise_adjuster.adjust_noise(
                hospital_id,
                risk_result['risk_score'],
                trend
            )
            
            # Store for next round
            self.client_risk_scores[hospital_id] = risk_result['risk_score']
            self.client_noise_levels[hospital_id] = noise_adjustment['new_noise']
    
    def _adjust_noise_levels(self):
        """
        Adjust noise levels based on risk scores (Adaptive Privacy Engine)
        """
        print("\n🎛️ Adjusting noise levels...")
        
        for hospital_id, risk in self.client_risk_scores.items():
            current_noise = self.client_noise_levels[hospital_id]
            
            if risk > HIGH_RISK_THRESHOLD:
                # High risk → increase noise
                new_noise = current_noise * (1 + NOISE_INCREASE_RATE)
                print(f"   Hospital {hospital_id}: HIGH RISK ({risk:.3f}) → "
                      f"Increase noise {current_noise:.3f} → {new_noise:.3f}")
            
            elif risk < LOW_RISK_THRESHOLD:
                # Low risk → decrease noise (improve accuracy)
                new_noise = current_noise * (1 - NOISE_DECREASE_RATE)
                print(f"   Hospital {hospital_id}: LOW RISK ({risk:.3f}) → "
                      f"Decrease noise {current_noise:.3f} → {new_noise:.3f}")
            
            else:
                # Medium risk → keep same
                new_noise = current_noise
                print(f"   Hospital {hospital_id}: MEDIUM RISK ({risk:.3f}) → "
                      f"Keep noise {current_noise:.3f}")
            
            # Update noise level (with bounds)
            self.client_noise_levels[hospital_id] = np.clip(new_noise, 0.1, 2.0)
    
    def _log_round_metrics(self, server_round, results):
        """
        Log metrics for dashboard
        """
        round_data = {
            "round": server_round,
            "clients": {}
        }
        
        for client, fit_res in results:
            hospital_id = chr(65 + int(client.cid))
            round_data["clients"][hospital_id] = {
                "risk_score": self.client_risk_scores.get(hospital_id, 0),
                "noise_level": self.client_noise_levels.get(hospital_id, 0),
                "train_acc": fit_res.metrics.get("train_acc", 0),
                "num_samples": fit_res.num_examples
            }
        
        self.round_metrics.append(round_data)
        
        # Write to privacy_ledger.json
        ledger = get_ledger()
        global_metrics = {}
        if results:
            # Extract global metrics if available
            avg_acc = np.mean([r[1].metrics.get("train_acc", 0) for r in results])
            global_metrics = {"avg_train_acc": float(avg_acc)}
        
        ledger.log_round(server_round, round_data["clients"], global_metrics)


if __name__ == "__main__":
    print("🧪 Testing Adaptive Privacy Strategy...\n")
    strategy = AdaptivePrivacyStrategy()
    print("\n✅ Strategy initialized successfully!")
    print(f"   Risk scores: {strategy.client_risk_scores}")
    print(f"   Noise levels: {strategy.client_noise_levels}")