"""
Visualization Plotting Functions
Helper functions for creating plots and visualizations
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


def plot_risk_scores_over_time(round_data: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot risk scores over training rounds
    
    Args:
        round_data: List of round data dicts with 'round' and 'clients' keys
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data per hospital
    hospitals = set()
    for round_info in round_data:
        hospitals.update(round_info.get("clients", {}).keys())
    
    hospitals = sorted(list(hospitals))
    
    for hospital_id in hospitals:
        rounds = []
        risk_scores = []
        
        for round_info in round_data:
            if hospital_id in round_info.get("clients", {}):
                rounds.append(round_info["round"])
                risk_scores.append(round_info["clients"][hospital_id].get("risk_score", 0))
        
        if rounds:
            ax.plot(rounds, risk_scores, marker='o', label=f'Hospital {hospital_id}', linewidth=2)
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Risk Score', fontsize=12)
    ax.set_title('Privacy Risk Scores Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_noise_levels_over_time(round_data: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot noise levels over training rounds
    
    Args:
        round_data: List of round data dicts
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    hospitals = set()
    for round_info in round_data:
        hospitals.update(round_info.get("clients", {}).keys())
    
    hospitals = sorted(list(hospitals))
    
    for hospital_id in hospitals:
        rounds = []
        noise_levels = []
        
        for round_info in round_data:
            if hospital_id in round_info.get("clients", {}):
                rounds.append(round_info["round"])
                noise_levels.append(round_info["clients"][hospital_id].get("noise_level", 0))
        
        if rounds:
            ax.plot(rounds, noise_levels, marker='s', label=f'Hospital {hospital_id}', linewidth=2)
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Noise Level', fontsize=12)
    ax.set_title('Adaptive Noise Levels Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_accuracy_over_time(round_data: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot training accuracy over rounds
    
    Args:
        round_data: List of round data dicts
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    hospitals = set()
    for round_info in round_data:
        hospitals.update(round_info.get("clients", {}).keys())
    
    hospitals = sorted(list(hospitals))
    
    for hospital_id in hospitals:
        rounds = []
        accuracies = []
        
        for round_info in round_data:
            if hospital_id in round_info.get("clients", {}):
                rounds.append(round_info["round"])
                accuracies.append(round_info["clients"][hospital_id].get("train_acc", 0))
        
        if rounds:
            ax.plot(rounds, accuracies, marker='o', label=f'Hospital {hospital_id}', linewidth=2)
    
    # Global accuracy if available
    global_rounds = []
    global_accs = []
    for round_info in round_data:
        if "global_metrics" in round_info and "avg_train_acc" in round_info["global_metrics"]:
            global_rounds.append(round_info["round"])
            global_accs.append(round_info["global_metrics"]["avg_train_acc"])
    
    if global_rounds:
        ax.plot(global_rounds, global_accs, marker='*', label='Global Average', 
                linewidth=3, linestyle='--', markersize=10)
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_risk_vs_noise(round_data: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    Scatter plot of risk scores vs noise levels
    
    Args:
        round_data: List of round data dicts
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    hospitals = set()
    for round_info in round_data:
        hospitals.update(round_info.get("clients", {}).keys())
    
    hospitals = sorted(list(hospitals))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for idx, hospital_id in enumerate(hospitals):
        risk_scores = []
        noise_levels = []
        
        for round_info in round_data:
            if hospital_id in round_info.get("clients", {}):
                client_data = round_info["clients"][hospital_id]
                risk_scores.append(client_data.get("risk_score", 0))
                noise_levels.append(client_data.get("noise_level", 0))
        
        if risk_scores:
            ax.scatter(risk_scores, noise_levels, label=f'Hospital {hospital_id}', 
                      s=100, alpha=0.6, color=colors[idx % len(colors)])
    
    ax.set_xlabel('Risk Score', fontsize=12)
    ax.set_ylabel('Noise Level', fontsize=12)
    ax.set_title('Risk Score vs Noise Level', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_privacy_utility_tradeoff(round_data: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot privacy-utility tradeoff (noise level vs accuracy)
    
    Args:
        round_data: List of round data dicts
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    hospitals = set()
    for round_info in round_data:
        hospitals.update(round_info.get("clients", {}).keys())
    
    hospitals = sorted(list(hospitals))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for idx, hospital_id in enumerate(hospitals):
        noise_levels = []
        accuracies = []
        
        for round_info in round_data:
            if hospital_id in round_info.get("clients", {}):
                client_data = round_info["clients"][hospital_id]
                noise_levels.append(client_data.get("noise_level", 0))
                accuracies.append(client_data.get("train_acc", 0))
        
        if noise_levels:
            ax.plot(noise_levels, accuracies, marker='o', label=f'Hospital {hospital_id}', 
                   linewidth=2, color=colors[idx % len(colors)], alpha=0.7)
    
    ax.set_xlabel('Noise Level (Privacy)', fontsize=12)
    ax.set_ylabel('Accuracy (Utility)', fontsize=12)
    ax.set_title('Privacy-Utility Tradeoff', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_summary_dashboard(round_data: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    Create a comprehensive dashboard with multiple subplots
    
    Args:
        round_data: List of round data dicts
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract hospital data
    hospitals = set()
    for round_info in round_data:
        hospitals.update(round_info.get("clients", {}).keys())
    hospitals = sorted(list(hospitals))
    
    # Plot 1: Risk scores
    ax1 = axes[0, 0]
    for hospital_id in hospitals:
        rounds = []
        risk_scores = []
        for round_info in round_data:
            if hospital_id in round_info.get("clients", {}):
                rounds.append(round_info["round"])
                risk_scores.append(round_info["clients"][hospital_id].get("risk_score", 0))
        if rounds:
            ax1.plot(rounds, risk_scores, marker='o', label=f'Hospital {hospital_id}')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Risk Score')
    ax1.set_title('Risk Scores Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Noise levels
    ax2 = axes[0, 1]
    for hospital_id in hospitals:
        rounds = []
        noise_levels = []
        for round_info in round_data:
            if hospital_id in round_info.get("clients", {}):
                rounds.append(round_info["round"])
                noise_levels.append(round_info["clients"][hospital_id].get("noise_level", 0))
        if rounds:
            ax2.plot(rounds, noise_levels, marker='s', label=f'Hospital {hospital_id}')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Noise Level')
    ax2.set_title('Noise Levels Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy
    ax3 = axes[1, 0]
    for hospital_id in hospitals:
        rounds = []
        accuracies = []
        for round_info in round_data:
            if hospital_id in round_info.get("clients", {}):
                rounds.append(round_info["round"])
                accuracies.append(round_info["clients"][hospital_id].get("train_acc", 0))
        if rounds:
            ax3.plot(rounds, accuracies, marker='o', label=f'Hospital {hospital_id}')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Training Accuracy Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Plot 4: Risk vs Noise
    ax4 = axes[1, 1]
    colors = ['red', 'blue', 'green']
    for idx, hospital_id in enumerate(hospitals):
        risk_scores = []
        noise_levels = []
        for round_info in round_data:
            if hospital_id in round_info.get("clients", {}):
                client_data = round_info["clients"][hospital_id]
                risk_scores.append(client_data.get("risk_score", 0))
                noise_levels.append(client_data.get("noise_level", 0))
        if risk_scores:
            ax4.scatter(risk_scores, noise_levels, label=f'Hospital {hospital_id}', 
                       s=100, alpha=0.6, color=colors[idx % len(colors)])
    ax4.set_xlabel('Risk Score')
    ax4.set_ylabel('Noise Level')
    ax4.set_title('Risk vs Noise')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 1])
    
    plt.suptitle('FLAIR-X Privacy Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("🧪 Testing visualization plots...\n")
    
    # Create dummy data
    dummy_data = [
        {
            "round": 1,
            "clients": {
                "A": {"risk_score": 0.6, "noise_level": 0.5, "train_acc": 0.7},
                "B": {"risk_score": 0.4, "noise_level": 0.4, "train_acc": 0.75},
                "C": {"risk_score": 0.5, "noise_level": 0.45, "train_acc": 0.72}
            }
        },
        {
            "round": 2,
            "clients": {
                "A": {"risk_score": 0.7, "noise_level": 0.6, "train_acc": 0.72},
                "B": {"risk_score": 0.3, "noise_level": 0.35, "train_acc": 0.78},
                "C": {"risk_score": 0.45, "noise_level": 0.4, "train_acc": 0.74}
            }
        }
    ]
    
    print("1. Creating risk scores plot...")
    plot_risk_scores_over_time(dummy_data)
    print("   ✅ Risk scores plot created")
    
    print("\n2. Creating noise levels plot...")
    plot_noise_levels_over_time(dummy_data)
    print("   ✅ Noise levels plot created")
    
    print("\n3. Creating accuracy plot...")
    plot_accuracy_over_time(dummy_data)
    print("   ✅ Accuracy plot created")
    
    print("\n4. Creating summary dashboard...")
    create_summary_dashboard(dummy_data)
    print("   ✅ Summary dashboard created")
    
    print("\n✅ All plots tested successfully!")
    print("   (Note: Figures are created but not displayed. Use in dashboard or save to file.)")

