"""
FLAIR-X Main Entry Point - Sequential Training (No Ray)
Runs federated learning with 3 hospitals sequentially
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from config import NUM_ROUNDS, NUM_CLIENTS, HOSPITAL_IDS
from federated.client import HospitalClient
from federated.model import get_model
from federated.utils import get_all_data_loaders, get_parameters, set_parameters
from ledger.logger import get_ledger


class SequentialFederatedLearning:
    """Sequential FL without Ray - more stable on Windows"""
    
    def __init__(self):
        self.clients = {}
        self.client_risk_scores = {h: 0.5 for h in HOSPITAL_IDS}
        self.client_noise_levels = {h: 0.5 for h in HOSPITAL_IDS}
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def create_clients(self):
        """Initialize all hospital clients"""
        print("\n🏗️ Creating hospital clients...\n")
        
        for hospital_id in HOSPITAL_IDS:
            print(f"📍 Setting up Hospital {hospital_id}...")
            data_loaders = get_all_data_loaders(hospital_id)
            model = get_model()
            self.clients[hospital_id] = HospitalClient(hospital_id, data_loaders, model)
            print()
    
    def aggregate_parameters(self, client_results):
        """Weighted average aggregation"""
        if not client_results:
            return None
        
        total_samples = sum(r['num_samples'] for r in client_results)
        
        # Initialize aggregated parameters with zeros
        num_params = len(client_results[0]['parameters'])
        aggregated = []
        
        for i in range(num_params):
            # Weighted sum
            weighted_param = sum(
                r['parameters'][i] * (r['num_samples'] / total_samples)
                for r in client_results
            )
            aggregated.append(weighted_param)
        
        return aggregated
    
    def compute_risk_score(self, parameters, metrics):
        """Compute privacy risk (simplified)"""
        # In practice, this would analyze gradient patterns
        # For demo: use training loss as proxy
        loss = metrics.get('train_loss', 1.0)
        
        # Higher loss = more risk (simplified assumption)
        risk = np.clip(loss / 3.0, 0.0, 1.0)
        
        # Add randomness to simulate real privacy auditing
        risk += np.random.normal(0, 0.05)
        risk = np.clip(risk, 0.0, 1.0)
        
        return float(risk)
    
    def adjust_noise_levels(self):
        """Adaptive privacy: adjust noise based on risk"""
        HIGH_RISK_THRESHOLD = 0.7
        LOW_RISK_THRESHOLD = 0.3
        NOISE_INCREASE_RATE = 0.1
        NOISE_DECREASE_RATE = 0.05
        
        print("\n🎛️ Adjusting noise levels...")
        
        for hospital_id, risk in self.client_risk_scores.items():
            current_noise = self.client_noise_levels[hospital_id]
            
            if risk > HIGH_RISK_THRESHOLD:
                new_noise = current_noise * (1 + NOISE_INCREASE_RATE)
                action = "Increase"
                status = "HIGH RISK"
            elif risk < LOW_RISK_THRESHOLD:
                new_noise = current_noise * (1 - NOISE_DECREASE_RATE)
                action = "Decrease"
                status = "LOW RISK"
            else:
                new_noise = current_noise
                action = "Keep"
                status = "MEDIUM RISK"
            
            # Bounds: [0.1, 2.0]
            new_noise = np.clip(new_noise, 0.1, 2.0)
            self.client_noise_levels[hospital_id] = new_noise
            
            print(f"   Hospital {hospital_id}: {status} ({risk:.3f}) → "
                  f"{action} noise {current_noise:.3f} → {new_noise:.3f}")
    
    def train(self):
        """Main training loop"""
        # Initialize global model
        global_model = get_model()
        global_params = get_parameters(global_model)
        
        print("\n" + "="*70)
        print("🌐 Starting Sequential Federated Training")
        print("="*70)
        
        for round_num in range(1, NUM_ROUNDS + 1):
            print(f"\n{'='*70}")
            print(f"📍 ROUND {round_num}/{NUM_ROUNDS}")
            print('='*70)
            
            # ========== TRAINING PHASE ==========
            print(f"\n🔄 Round {round_num} - Configured clients:")
            for hospital_id in HOSPITAL_IDS:
                noise = self.client_noise_levels[hospital_id]
                risk = self.client_risk_scores[hospital_id]
                print(f"   Hospital {hospital_id}: noise={noise:.3f}, risk={risk:.3f}")
            
            client_results = []
            
            for hospital_id in HOSPITAL_IDS:
                client = self.clients[hospital_id]
                noise = self.client_noise_levels[hospital_id]
                
                print(f"\n🏋️ Hospital {hospital_id} - Starting local training...")
                
                try:
                    config = {
                        "noise_multiplier": noise,
                        "server_round": round_num
                    }
                    
                    updated_params, num_samples, metrics = client.fit(global_params, config)
                    
                    client_results.append({
                        'hospital_id': hospital_id,
                        'parameters': updated_params,
                        'num_samples': num_samples,
                        'metrics': metrics
                    })
                    
                    print(f"✅ Hospital {hospital_id} - Training complete")
                    print(f"   Loss: {metrics['train_loss']:.4f}, Acc: {metrics['train_acc']:.4f}")
                    
                except Exception as e:
                    print(f"❌ Hospital {hospital_id} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # ========== AGGREGATION PHASE ==========
            print(f"\n📊 Round {round_num} - Aggregating results...")
            
            if not client_results:
                print("   ⚠️ All clients failed - skipping round")
                continue
            
            print(f"   ✅ Received {len(client_results)} client updates")
            
            # Aggregate parameters
            global_params = self.aggregate_parameters(client_results)
            
            # Update risk scores
            for result in client_results:
                hospital_id = result['hospital_id']
                risk = self.compute_risk_score(result['parameters'], result['metrics'])
                self.client_risk_scores[hospital_id] = risk
            
            # Adjust noise levels
            self.adjust_noise_levels()
            
            # ========== EVALUATION PHASE ==========
            print(f"\n📊 Round {round_num} - Evaluating global model...")
            
            val_losses = []
            val_accs = []
            
            for hospital_id in HOSPITAL_IDS:
                client = self.clients[hospital_id]
                
                try:
                    val_loss, num_samples, metrics = client.evaluate(global_params, {})
                    val_losses.append(val_loss)
                    val_accs.append(metrics['val_acc'])
                    
                    print(f"   Hospital {hospital_id}: Loss={val_loss:.4f}, Acc={metrics['val_acc']:.4f}")
                
                except Exception as e:
                    print(f"   ⚠️ Hospital {hospital_id} evaluation failed: {e}")
                    continue
            
            # Compute global metrics
            if val_losses:
                avg_loss = np.mean(val_losses)
                avg_acc = np.mean(val_accs)
                
                self.history['val_loss'].append(avg_loss)
                self.history['val_acc'].append(avg_acc)
                
                print(f"\n   🌐 Global Metrics:")
                print(f"      Val Loss: {avg_loss:.4f}")
                print(f"      Val Acc: {avg_acc:.4f}")

            # ========== LEDGER LOGGING ==========
            try:
                ledger_clients = {}
                for result in client_results:
                    hid = result['hospital_id']
                    ledger_clients[hid] = {
                        "risk_score": float(self.client_risk_scores.get(hid, 0.0)),
                        "noise_level": float(self.client_noise_levels.get(hid, 0.0)),
                        "train_acc": float(result['metrics'].get('train_acc', 0.0)),
                        "train_loss": float(result['metrics'].get('train_loss', 0.0)),
                        "num_samples": int(result.get('num_samples', 0))
                    }

                global_metrics = {}
                if val_losses:
                    global_metrics = {
                        "val_loss": float(avg_loss),
                        "val_acc": float(avg_acc)
                    }

                ledger = get_ledger()
                ledger.log_round(round_num, ledger_clients, global_metrics)
            except Exception as e:
                print(f"   ⚠️ Failed to write ledger for round {round_num}: {e}")
        
        return global_params


def main():
    """Main entry point"""
    print("="*70)
    print("🚀 FLAIR-X: Federated Learning with Adaptive Privacy")
    print("   (Sequential Training Mode)")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"   Number of clients (hospitals): {NUM_CLIENTS}")
    print(f"   Number of rounds: {NUM_ROUNDS}")
    print(f"   Hospital IDs: {HOSPITAL_IDS}")
    print("\n" + "="*70)
    
    try:
        # Create FL system
        fl_system = SequentialFederatedLearning()
        
        # Initialize clients
        fl_system.create_clients()
        
        # Train
        final_params = fl_system.train()
        
        # Print final results
        print("\n" + "="*70)
        print("✅ Federated Learning Complete!")
        print("="*70)
        
        print("\n📊 Final Results:")
        print(f"   Total rounds completed: {NUM_ROUNDS}")
        
        if fl_system.history['val_acc']:
            print(f"   Best Val Accuracy: {max(fl_system.history['val_acc']):.4f}")
            print(f"   Final Val Accuracy: {fl_system.history['val_acc'][-1]:.4f}")
        
        print("\n🔒 Final Privacy Metrics:")
        for hospital_id in HOSPITAL_IDS:
            risk = fl_system.client_risk_scores[hospital_id]
            noise = fl_system.client_noise_levels[hospital_id]
            print(f"   Hospital {hospital_id}:")
            print(f"      Risk Score: {risk:.4f}")
            print(f"      Noise Level: {noise:.4f}")
        
        print("\n" + "="*70)
        print("🎉 Simulation finished successfully!")
        print("   Next steps:")
        print("   1. Check outputs/logs/ for training logs")
        print("   2. Run dashboard: streamlit run src/visualization/dashboard.py")
        print("="*70)
        
        return fl_system
    
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    fl_system = main()
    
    if fl_system is not None:
        print("\n✅ Training complete! System ready.")
    else:
        print("\n❌ Training failed. Check errors above.")