"""
Privacy Ledger Logger
Logs all privacy-related events and metrics to a JSON ledger
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import LEDGER_PATH


class PrivacyLedger:
    """
    Privacy ledger for tracking all privacy events and metrics
    Maintains an append-only log of privacy-related activities
    """
    
    def __init__(self, ledger_path: Optional[Path] = None):
        self.ledger_path = ledger_path or LEDGER_PATH
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize ledger if it doesn't exist
        if not self.ledger_path.exists() or self.ledger_path.stat().st_size == 0:
            self._initialize_ledger()
    
    def _initialize_ledger(self):
        """Initialize the ledger with a base structure"""
        initial_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "description": "FLAIR-X Privacy Ledger - Append-only log of privacy events"
            },
            "rounds": []
        }
        self._write_ledger(initial_data)
    
    def _read_ledger(self) -> Dict[str, Any]:
        """Read the current ledger"""
        try:
            if not self.ledger_path.exists() or self.ledger_path.stat().st_size == 0:
                self._initialize_ledger()
            
            with open(self.ledger_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # If corrupted, reinitialize
            self._initialize_ledger()
            return self._read_ledger()
    
    def _write_ledger(self, data: Dict[str, Any]):
        """Write data to the ledger"""
        with open(self.ledger_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def log_round(
        self,
        round_num: int,
        clients: Dict[str, Dict[str, Any]],
        global_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Log a federated learning round
        
        Args:
            round_num: Round number
            clients: Dict mapping hospital_id to their metrics
            global_metrics: Global aggregated metrics
        """
        ledger = self._read_ledger()
        
        round_entry = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "clients": clients,
            "global_metrics": global_metrics or {}
        }
        
        # Append to rounds list
        rounds = ledger.get("rounds", [])
        
        # Update if round already exists, else append
        existing_idx = None
        for idx, r in enumerate(rounds):
            if r.get("round") == round_num:
                existing_idx = idx
                break
        
        if existing_idx is not None:
            rounds[existing_idx] = round_entry
        else:
            rounds.append(round_entry)
        
        ledger["rounds"] = rounds
        ledger["metadata"]["last_updated"] = datetime.now().isoformat()
        
        self._write_ledger(ledger)
    
    def log_privacy_event(
        self,
        event_type: str,
        hospital_id: str,
        details: Dict[str, Any]
    ):
        """
        Log a specific privacy event (e.g., noise adjustment, risk assessment)
        
        Args:
            event_type: Type of event (e.g., "noise_adjustment", "risk_assessment")
            hospital_id: Hospital identifier
            details: Event-specific details
        """
        ledger = self._read_ledger()
        
        # Get or create events list
        if "events" not in ledger:
            ledger["events"] = []
        
        event_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "hospital_id": hospital_id,
            "details": details
        }
        
        ledger["events"].append(event_entry)
        ledger["metadata"]["last_updated"] = datetime.now().isoformat()
        
        self._write_ledger(ledger)
    
    def get_round_history(self, max_rounds: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get round history, optionally limited to last N rounds"""
        ledger = self._read_ledger()
        rounds = ledger.get("rounds", [])
        
        if max_rounds:
            return rounds[-max_rounds:]
        return rounds
    
    def get_client_history(self, hospital_id: str) -> List[Dict[str, Any]]:
        """Get history for a specific hospital"""
        rounds = self.get_round_history()
        
        client_history = []
        for round_data in rounds:
            if hospital_id in round_data.get("clients", {}):
                client_history.append({
                    "round": round_data["round"],
                    "timestamp": round_data["timestamp"],
                    "metrics": round_data["clients"][hospital_id]
                })
        
        return client_history
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the ledger"""
        ledger = self._read_ledger()
        rounds = ledger.get("rounds", [])
        
        if not rounds:
            return {
                "total_rounds": 0,
                "created_at": ledger["metadata"].get("created_at"),
                "last_updated": ledger["metadata"].get("last_updated")
            }
        
        # Aggregate statistics
        all_risk_scores = []
        all_noise_levels = []
        all_train_accs = []
        
        for round_data in rounds:
            for hospital_id, client_data in round_data.get("clients", {}).items():
                if "risk_score" in client_data:
                    all_risk_scores.append(client_data["risk_score"])
                if "noise_level" in client_data:
                    all_noise_levels.append(client_data["noise_level"])
                if "train_acc" in client_data:
                    all_train_accs.append(client_data["train_acc"])
        
        summary = {
            "total_rounds": len(rounds),
            "created_at": ledger["metadata"].get("created_at"),
            "last_updated": ledger["metadata"].get("last_updated"),
            "total_events": len(ledger.get("events", [])),
            "statistics": {
                "risk_scores": {
                    "mean": sum(all_risk_scores) / len(all_risk_scores) if all_risk_scores else 0,
                    "min": min(all_risk_scores) if all_risk_scores else 0,
                    "max": max(all_risk_scores) if all_risk_scores else 0
                },
                "noise_levels": {
                    "mean": sum(all_noise_levels) / len(all_noise_levels) if all_noise_levels else 0,
                    "min": min(all_noise_levels) if all_noise_levels else 0,
                    "max": max(all_noise_levels) if all_noise_levels else 0
                },
                "train_accuracies": {
                    "mean": sum(all_train_accs) / len(all_train_accs) if all_train_accs else 0,
                    "min": min(all_train_accs) if all_train_accs else 0,
                    "max": max(all_train_accs) if all_train_accs else 0
                }
            }
        }
        
        return summary


# Global instance
_ledger_instance = None

def get_ledger() -> PrivacyLedger:
    """Get or create the global ledger instance"""
    global _ledger_instance
    if _ledger_instance is None:
        _ledger_instance = PrivacyLedger()
    return _ledger_instance


if __name__ == "__main__":
    # Test the ledger
    print("🧪 Testing Privacy Ledger...\n")
    
    ledger = PrivacyLedger()
    
    # Test logging a round
    print("1. Logging a test round...")
    ledger.log_round(
        round_num=1,
        clients={
            "A": {
                "risk_score": 0.65,
                "noise_level": 0.5,
                "train_acc": 0.75,
                "num_samples": 100
            },
            "B": {
                "risk_score": 0.45,
                "noise_level": 0.4,
                "train_acc": 0.80,
                "num_samples": 150
            }
        },
        global_metrics={"val_acc": 0.78}
    )
    print("✅ Round logged")
    
    # Test logging an event
    print("\n2. Logging a privacy event...")
    ledger.log_privacy_event(
        event_type="noise_adjustment",
        hospital_id="A",
        details={
            "old_noise": 0.5,
            "new_noise": 0.6,
            "reason": "high_risk"
        }
    )
    print("✅ Event logged")
    
    # Test getting summary
    print("\n3. Getting ledger summary...")
    summary = ledger.get_summary()
    print(f"✅ Summary: {json.dumps(summary, indent=2)}")
    
    print("\n✅ All tests passed!")

