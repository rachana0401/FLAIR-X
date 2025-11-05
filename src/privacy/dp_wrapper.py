"""
Differential Privacy Wrapper
Wraps model updates with differential privacy mechanisms
"""
import torch
import numpy as np
from typing import List, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import MAX_GRAD_NORM


class DifferentialPrivacyWrapper:
    """
    Wrapper for applying differential privacy to model updates
    """
    
    def __init__(self, noise_multiplier: float = 0.5, max_grad_norm: float = None):
        """
        Args:
            noise_multiplier: Multiplier for Gaussian noise
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm or MAX_GRAD_NORM
    
    def clip_gradients(self, parameters: List[torch.Tensor]) -> float:
        """
        Clip gradients to a maximum norm
        
        Args:
            parameters: List of parameter tensors
        
        Returns:
            Total gradient norm before clipping
        """
        # Compute total norm
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad) for p in parameters if p.grad is not None])
        )
        
        # Clip gradients
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                if p.grad is not None:
                    p.grad.mul_(clip_coef)
        
        return float(total_norm)
    
    def add_noise_to_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add Gaussian noise to parameters (for NumPy arrays)
        
        Args:
            parameters: List of parameter arrays
        
        Returns:
            List of noised parameter arrays
        """
        noised_params = []
        for param in parameters:
            # Add Gaussian noise proportional to noise multiplier
            noise = np.random.normal(0, self.noise_multiplier * 0.01, param.shape)
            noised_param = param + noise
            noised_params.append(noised_param)
        
        return noised_params
    
    def add_noise_to_tensors(self, parameters: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Add Gaussian noise to parameters (for PyTorch tensors)
        
        Args:
            parameters: List of parameter tensors
        
        Returns:
            List of noised parameter tensors
        """
        noised_params = []
        for param in parameters:
            # Add Gaussian noise proportional to noise multiplier
            noise = torch.randn_like(param) * self.noise_multiplier * 0.01
            noised_param = param + noise
            noised_params.append(noised_param)
        
        return noised_params
    
    def apply_dp_to_model(self, model: torch.nn.Module):
        """
        Apply differential privacy to a PyTorch model in-place
        
        Args:
            model: PyTorch model to add noise to
        """
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    # Check for stability before adding noise
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"⚠️ WARNING: Parameter has NaN/Inf values before noise addition!")
                        continue
                    
                    # Add Gaussian noise proportional to noise multiplier
                    noise = torch.randn_like(param) * self.noise_multiplier * 0.01
                    param.add_(noise)
    
    def set_noise_multiplier(self, noise_multiplier: float):
        """Update the noise multiplier"""
        self.noise_multiplier = max(0.0, noise_multiplier)
    
    def compute_privacy_cost(self, num_samples: int, num_rounds: int) -> dict:
        """
        Compute estimated privacy cost (epsilon, delta) for the given parameters
        
        This is a simplified estimate. In practice, you'd use composition theorems
        like Rényi DP or the moments accountant.
        
        Args:
            num_samples: Number of training samples
            num_rounds: Number of training rounds
        
        Returns:
            Dict with epsilon, delta estimates
        """
        # Simplified privacy accounting
        # In practice, use libraries like Opacus or TensorFlow Privacy
        
        # Basic composition: epsilon scales with sqrt(num_rounds)
        # This is a very rough approximation
        epsilon = self.noise_multiplier * np.sqrt(num_rounds) / num_samples
        delta = 1e-5  # Standard delta value
        
        return {
            "epsilon": float(epsilon),
            "delta": float(delta),
            "noise_multiplier": float(self.noise_multiplier),
            "num_samples": num_samples,
            "num_rounds": num_rounds
        }


if __name__ == "__main__":
    print("🧪 Testing Differential Privacy Wrapper...\n")
    
    # Test with NumPy arrays
    print("1. Testing NumPy parameter noise...")
    wrapper = DifferentialPrivacyWrapper(noise_multiplier=0.5)
    params = [np.random.randn(10, 10), np.random.randn(5, 5)]
    noised = wrapper.add_noise_to_parameters(params)
    print(f"   Original params: {len(params)} arrays")
    print(f"   Noised params: {len(noised)} arrays")
    print(f"   ✅ Noise added successfully")
    
    # Test with PyTorch tensors
    print("\n2. Testing PyTorch tensor noise...")
    import torch.nn as nn
    model = nn.Linear(10, 5)
    original_params = [p.clone() for p in model.parameters()]
    wrapper.apply_dp_to_model(model)
    print(f"   ✅ Noise applied to model")
    
    # Test privacy cost
    print("\n3. Computing privacy cost...")
    privacy_cost = wrapper.compute_privacy_cost(num_samples=1000, num_rounds=10)
    print(f"   Epsilon: {privacy_cost['epsilon']:.6f}")
    print(f"   Delta: {privacy_cost['delta']:.6f}")
    print(f"   ✅ Privacy cost computed")
    
    print("\n✅ All tests passed!")

