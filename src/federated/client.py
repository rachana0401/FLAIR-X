"""
Flower Client for FLAIR-X
Represents a single hospital training locally with differential privacy
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import traceback
import numpy as np

from config import (
    NUM_EPOCHS_PER_ROUND, 
    LEARNING_RATE, 
    DEVICE,
    INITIAL_NOISE_MULTIPLIER
)
from federated.model import get_model
from federated.utils import (
    get_all_data_loaders,
    get_parameters,
    set_parameters,
    train_one_epoch,
    evaluate
)


class HospitalClient(fl.client.NumPyClient):
    """
    Flower client representing a hospital in federated learning
    """
    
    def __init__(self, hospital_id, data_loaders, model):
        """
        Args:
            hospital_id: "A", "B", or "C"
            data_loaders: dict with 'train', 'val', 'test' loaders
            model: PyTorch model
        """
        self.hospital_id = hospital_id
        self.data_loaders = data_loaders
        self.model = model
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # Privacy tracking
        self.current_noise_multiplier = INITIAL_NOISE_MULTIPLIER
        
        print(f"🏥 Hospital {hospital_id} initialized")
        print(f"   Device: {self.device}")
        print(f"   Train samples: {len(self.data_loaders['train'].dataset)}")
        print(f"   Val samples: {len(self.data_loaders['val'].dataset)}")
        
        # Verify data loaders are not empty
        for split in ['train', 'val']:
            loader = self.data_loaders[split]
            print(f"   {split} batches: {len(loader)}")
            if len(loader) == 0:
                raise ValueError(f"Empty {split} data loader for Hospital {hospital_id}!")
    
    def get_parameters(self, config):
        """
        Return current model parameters as NumPy arrays
        Called by Flower server
        """
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        """
        Train model on local data with comprehensive error handling
        """
        print(f"\n🏋️ Hospital {self.hospital_id} - Starting local training...")
        sys.stdout.flush()

        try:
            # Step 1: Set parameters
            print(f"<<< DEBUG {self.hospital_id}: Setting parameters...")
            sys.stdout.flush()
            set_parameters(self.model, parameters)
            print(f"<<< DEBUG {self.hospital_id}: Parameters set successfully")
            sys.stdout.flush()

            # Step 2: Update noise multiplier
            if "noise_multiplier" in config:
                self.current_noise_multiplier = config["noise_multiplier"]
                print(f"   Noise multiplier: {self.current_noise_multiplier:.3f}")
                sys.stdout.flush()

            # Step 3: Training loop
            print(f"<<< DEBUG {self.hospital_id}: Starting {NUM_EPOCHS_PER_ROUND} epochs...")
            print(f"<<< DEBUG {self.hospital_id}: Train batches: {len(self.data_loaders['train'])}")
            sys.stdout.flush()
            
            train_losses = []
            train_accs = []
            
            for epoch in range(NUM_EPOCHS_PER_ROUND):
                try:
                    print(f"<<< DEBUG {self.hospital_id}: Starting epoch {epoch+1}/{NUM_EPOCHS_PER_ROUND}")
                    sys.stdout.flush()
                    
                    train_loss, train_acc = train_one_epoch(
                        self.model,
                        self.data_loaders['train'],
                        self.criterion,
                        self.optimizer,
                        self.device
                    )
                    
                    print(f"   Epoch {epoch+1}/{NUM_EPOCHS_PER_ROUND} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                    sys.stdout.flush()
                    
                    # Check for NaN/Inf
                    if np.isnan(train_loss) or np.isinf(train_loss):
                        raise ValueError(f"Training produced NaN/Inf loss: {train_loss}")
                    
                    train_losses.append(train_loss)
                    train_accs.append(train_acc)
                    
                except Exception as epoch_error:
                    print(f"❌ ERROR in epoch {epoch+1} for Hospital {self.hospital_id}: {epoch_error}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()
                    raise

            # Step 4: Apply DP noise
            print(f"<<< DEBUG {self.hospital_id}: Applying DP noise...")
            sys.stdout.flush()
            self._add_dp_noise()
            print(f"<<< DEBUG {self.hospital_id}: DP noise applied")
            sys.stdout.flush()

            # Step 5: Get parameters
            print(f"<<< DEBUG {self.hospital_id}: Extracting parameters...")
            sys.stdout.flush()
            updated_parameters = get_parameters(self.model)
            print(f"<<< DEBUG {self.hospital_id}: Parameters extracted - {len(updated_parameters)} arrays")
            sys.stdout.flush()

            # Step 6: Prepare return values
            num_samples = len(self.data_loaders['train'].dataset)
            
            metrics = {
                "hospital_id": self.hospital_id,
                "train_loss": float(train_losses[-1]),
                "train_acc": float(train_accs[-1]),
                "noise_multiplier": float(self.current_noise_multiplier)
            }

            print(f"✅ Hospital {self.hospital_id} - Training complete, returning {num_samples} samples")
            sys.stdout.flush()
            
            return updated_parameters, num_samples, metrics

        except Exception as e:
            print(f"\n❌ FATAL ERROR IN FIT FOR HOSPITAL {self.hospital_id}: {type(e).__name__}: {e}", file=sys.stderr)
            sys.stderr.flush()
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise

    def evaluate(self, parameters, config):
        """
        Evaluate global model on local validation data
        """
        print(f"📊 Hospital {self.hospital_id} - Evaluating...")
        
        # Set global model parameters
        set_parameters(self.model, parameters)
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate(
            self.model,
            self.data_loaders['val'],
            self.criterion,
            self.device
        )
        
        num_samples = len(self.data_loaders['val'].dataset)
        
        metrics = {
            "hospital_id": self.hospital_id,
            "val_acc": val_acc
        }
        
        print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        return val_loss, num_samples, metrics
   
    def _add_dp_noise(self):
        """
        Add Gaussian noise to model parameters (Differential Privacy)
        """
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    # Check for stability before adding noise
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"🚨 WARNING: Parameter has NaN/Inf values before noise addition!", file=sys.stderr)
                        sys.stderr.flush()

                    # Add Gaussian noise proportional to noise multiplier
                    noise = torch.randn_like(param) * self.current_noise_multiplier * 0.01
                    param.add_(noise)


def create_client(hospital_id):
    """
    Factory function to create a Flower client for a hospital
    """
    print(f"🏗️ Creating client for Hospital {hospital_id}...")
    
    # Load data
    data_loaders = get_all_data_loaders(hospital_id)
    
    # Create model
    model = get_model()
    
    # Create NumPyClient
    numpy_client = HospitalClient(hospital_id, data_loaders, model)
    
    print(f"✅ Client for Hospital {hospital_id} created successfully")
    
    # Convert to Client (fixes deprecation warning)
    return numpy_client.to_client()

# Test code (only runs when file is executed directly)
if __name__ == "__main__":
    print("🧪 Testing Hospital Client...\n")
    
    try:
        # Test client creation
        print("=" * 50)
        print("Test 1: Creating client for Hospital A")
        print("=" * 50)
        client = create_client("A")
        print("\n✅ Client created successfully!")
        
        # Test getting parameters
        print("\n" + "=" * 50)
        print("Test 2: Extracting parameters")
        print("=" * 50)
        params = client.get_parameters({})
        print(f"✅ Parameters extracted: {len(params)} arrays")
        
        # Test fit (with dummy parameters)
        print("\n" + "=" * 50)
        print("Test 3: Testing training round (fit)")
        print("=" * 50)
        updated_params, num_samples, metrics = client.fit(params, {"noise_multiplier": 0.5})
        print(f"\n✅ Training complete!")
        print(f"   Samples: {num_samples}")
        print(f"   Metrics: {metrics}")
        print(f"   Returned params: {len(updated_params)} arrays")
        
        # Test evaluate
        print("\n" + "=" * 50)
        print("Test 4: Testing evaluation")
        print("=" * 50)
        loss, num_val_samples, eval_metrics = client.evaluate(params, {})
        print(f"\n✅ Evaluation complete!")
        print(f"   Loss: {loss:.4f}")
        print(f"   Val samples: {num_val_samples}")
        print(f"   Metrics: {eval_metrics}")
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        traceback.print_exc()