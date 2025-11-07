"""
Visualization Plotting Functions with Animation Support
Helper functions for creating plots, visualizations, and animations
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


def plot_risk_scores_over_time(round_data: List[Dict[str, Any]], save_path: Optional[str] = None):
    """Plot risk scores over training rounds"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
    """Plot noise levels over training rounds"""
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
    """Plot training accuracy over rounds"""
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
    """Scatter plot of risk scores vs noise levels"""
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
    """Plot privacy-utility tradeoff (noise level vs accuracy)"""
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
    """Create a comprehensive dashboard with multiple subplots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
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
            ax1.plot(rounds, risk_scores, marker='o', label=f'Hospital {hospital_id}', linewidth=2.5)
    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Risk Score', fontsize=11)
    ax1.set_title('Risk Scores Over Time', fontsize=12, fontweight='bold')
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
            ax2.plot(rounds, noise_levels, marker='s', label=f'Hospital {hospital_id}', linewidth=2.5)
    ax2.set_xlabel('Round', fontsize=11)
    ax2.set_ylabel('Noise Level', fontsize=11)
    ax2.set_title('Noise Levels Over Time', fontsize=12, fontweight='bold')
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
            ax3.plot(rounds, accuracies, marker='o', label=f'Hospital {hospital_id}', linewidth=2.5)
    ax3.set_xlabel('Round', fontsize=11)
    ax3.set_ylabel('Accuracy', fontsize=11)
    ax3.set_title('Training Accuracy Over Time', fontsize=12, fontweight='bold')
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
    ax4.set_xlabel('Risk Score', fontsize=11)
    ax4.set_ylabel('Noise Level', fontsize=11)
    ax4.set_title('Risk vs Noise', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 1])
    
    plt.suptitle('FLAIR-X Privacy Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ==================== ANIMATION FUNCTIONS ====================

def create_animated_dashboard(round_data: List[Dict[str, Any]], 
                              save_path: Optional[str] = None,
                              fps: int = 2,
                              duration_per_round: float = 1.0):
    """
    Create an animated time-lapse dashboard showing training progress
    
    Args:
        round_data: List of round data dicts
        save_path: Path to save animation (e.g., 'dashboard.gif' or 'dashboard.mp4')
        fps: Frames per second
        duration_per_round: Duration to show each round (in seconds)
    
    Returns:
        animation object
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    hospitals = set()
    for round_info in round_data:
        hospitals.update(round_info.get("clients", {}).keys())
    hospitals = sorted(list(hospitals))
    colors = ['red', 'blue', 'green']
    
    # Initialize plots
    lines_risk = {}
    lines_noise = {}
    lines_acc = {}
    scatters = {}
    
    ax1 = axes[0, 0]  # Risk
    ax2 = axes[0, 1]  # Noise
    ax3 = axes[1, 0]  # Accuracy
    ax4 = axes[1, 1]  # Risk vs Noise
    
    for idx, hospital_id in enumerate(hospitals):
        lines_risk[hospital_id], = ax1.plot([], [], marker='o', label=f'Hospital {hospital_id}', 
                                            linewidth=2.5, color=colors[idx])
        lines_noise[hospital_id], = ax2.plot([], [], marker='s', label=f'Hospital {hospital_id}', 
                                             linewidth=2.5, color=colors[idx])
        lines_acc[hospital_id], = ax3.plot([], [], marker='o', label=f'Hospital {hospital_id}', 
                                           linewidth=2.5, color=colors[idx])
        scatters[hospital_id] = ax4.scatter([], [], label=f'Hospital {hospital_id}', 
                                           s=100, alpha=0.6, color=colors[idx])
    
    # Configure axes
    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Risk Score', fontsize=11)
    ax1.set_title('Risk Scores Over Time', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, len(round_data) + 1])
    
    ax2.set_xlabel('Round', fontsize=11)
    ax2.set_ylabel('Noise Level', fontsize=11)
    ax2.set_title('Noise Levels Over Time', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, len(round_data) + 1])
    
    ax3.set_xlabel('Round', fontsize=11)
    ax3.set_ylabel('Accuracy', fontsize=11)
    ax3.set_title('Training Accuracy Over Time', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    ax3.set_xlim([0, len(round_data) + 1])
    
    ax4.set_xlabel('Risk Score', fontsize=11)
    ax4.set_ylabel('Noise Level', fontsize=11)
    ax4.set_title('Risk vs Noise', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 1])
    
    plt.suptitle('FLAIR-X Privacy Dashboard - Training Progress', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Animation function
    def animate(frame):
        current_round = frame + 1
        
        for hospital_id in hospitals:
            rounds = []
            risk_scores = []
            noise_levels = []
            accuracies = []
            
            for round_info in round_data[:current_round]:
                if hospital_id in round_info.get("clients", {}):
                    rounds.append(round_info["round"])
                    client_data = round_info["clients"][hospital_id]
                    risk_scores.append(client_data.get("risk_score", 0))
                    noise_levels.append(client_data.get("noise_level", 0))
                    accuracies.append(client_data.get("train_acc", 0))
            
            # Update line plots
            if rounds:
                lines_risk[hospital_id].set_data(rounds, risk_scores)
                lines_noise[hospital_id].set_data(rounds, noise_levels)
                lines_acc[hospital_id].set_data(rounds, accuracies)
                
                # Update scatter
                scatters[hospital_id].set_offsets(np.c_[risk_scores, noise_levels])
        
        # Update title with current round
        plt.suptitle(f'FLAIR-X Privacy Dashboard - Round {current_round}/{len(round_data)}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        return list(lines_risk.values()) + list(lines_noise.values()) + list(lines_acc.values()) + list(scatters.values())
    
    # Create animation
    frames = len(round_data)
    interval = duration_per_round * 1000  # Convert to milliseconds
    
    anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                  interval=interval, blit=False, repeat=True)
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps, dpi=100)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=fps, dpi=100)
        print(f"✅ Animation saved to {save_path}")
    
    return anim


def create_round_summary_table(round_data: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    Create a summary table showing key metrics per round
    
    Args:
        round_data: List of round data dicts
        save_path: Optional path to save the figure
    """
    # Calculate average metrics per round
    summary_data = []
    
    for round_info in round_data:
        clients = round_info.get("clients", {})
        if not clients:
            continue
        
        avg_risk = np.mean([c.get("risk_score", 0) for c in clients.values()])
        avg_noise = np.mean([c.get("noise_level", 0) for c in clients.values()])
        avg_acc = np.mean([c.get("train_acc", 0) for c in clients.values()])
        
        summary_data.append({
            'Round': round_info["round"],
            'Avg Risk': f"{avg_risk:.2f}",
            'Avg Noise': f"{avg_noise:.2f}",
            'Accuracy': f"{avg_acc:.2f}"
        })
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(10, len(summary_data) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = [[d['Round'], d['Avg Risk'], d['Avg Noise'], d['Accuracy']] 
                  for d in summary_data]
    
    table = ax.table(cellText=table_data,
                    colLabels=['Round', 'Avg Risk', 'Avg Noise', 'Accuracy'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2, 0.3, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Simulated Round Highlights', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("🧪 Testing visualization plots with animations...\n")
    
    # Create dummy data
    dummy_data = [
        {"round": 1, "clients": {"A": {"risk_score": 0.72, "noise_level": 0.80, "train_acc": 0.65},
                                 "B": {"risk_score": 0.68, "noise_level": 0.75, "train_acc": 0.67},
                                 "C": {"risk_score": 0.75, "noise_level": 0.82, "train_acc": 0.63}}},
        {"round": 2, "clients": {"A": {"risk_score": 0.53, "noise_level": 0.60, "train_acc": 0.72},
                                 "B": {"risk_score": 0.50, "noise_level": 0.58, "train_acc": 0.74},
                                 "C": {"risk_score": 0.56, "noise_level": 0.62, "train_acc": 0.70}}},
        {"round": 3, "clients": {"A": {"risk_score": 0.39, "noise_level": 0.50, "train_acc": 0.79},
                                 "B": {"risk_score": 0.35, "noise_level": 0.48, "train_acc": 0.81},
                                 "C": {"risk_score": 0.42, "noise_level": 0.52, "train_acc": 0.77}}},
        {"round": 4, "clients": {"A": {"risk_score": 0.28, "noise_level": 0.40, "train_acc": 0.83},
                                 "B": {"risk_score": 0.25, "noise_level": 0.38, "train_acc": 0.85},
                                 "C": {"risk_score": 0.31, "noise_level": 0.42, "train_acc": 0.81}}},
    ]
    
    print("1. Creating round summary table...")
    create_round_summary_table(dummy_data, save_path="round_summary.png")
    print("   ✅ Round summary table created and saved as 'round_summary.png'")
    
    print("\n2. Creating animated dashboard...")
    print("   (This will take a moment...)")
    anim = create_animated_dashboard(dummy_data, save_path="dashboard_animation.gif", 
                                    fps=1, duration_per_round=1.0)
    print("   ✅ Animated dashboard saved as 'dashboard_animation.gif'")
    
    print("\n3. Creating static summary dashboard...")
    create_summary_dashboard(dummy_data, save_path="static_dashboard.png")
    print("   ✅ Static dashboard saved as 'static_dashboard.png'")
    
    print("\n✅ All visualizations created successfully!")
    print("\nGenerated files:")
    print("   - round_summary.png (table)")
    print("   - dashboard_animation.gif (animated)")
    print("   - static_dashboard.png (static)")