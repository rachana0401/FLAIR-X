"""
Streamlit Dashboard for FLAIR-X
Interactive dashboard for visualizing federated learning and privacy metrics
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from ledger.logger import get_ledger
from visualization.plots import (
    plot_risk_scores_over_time,
    plot_noise_levels_over_time,
    plot_accuracy_over_time,
    plot_risk_vs_noise,
    plot_privacy_utility_tradeoff,
    create_summary_dashboard
)
from config import LEDGER_PATH


def load_ledger_data():
    """Load data from the privacy ledger"""
    try:
        ledger = get_ledger()
        rounds = ledger.get_round_history()
        summary = ledger.get_summary()
        return rounds, summary
    except Exception as e:
        st.error(f"Error loading ledger data: {e}")
        return [], {}


def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="FLAIR-X Privacy Dashboard",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title
    st.title("🔒 FLAIR-X Privacy Dashboard")
    st.markdown("### Federated Learning with Adaptive Privacy")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Privacy Metrics", "Training Metrics", "Risk Analysis", "Privacy Ledger"]
    )
    
    # Load data
    rounds, summary = load_ledger_data()
    
    if not rounds:
        st.warning("⚠️ No data available. Please run federated learning training first.")
        st.info("Run: `python src/main.py` to start training")
        return
    
    # Main content based on selected page
    if page == "Overview":
        show_overview(rounds, summary)
    elif page == "Privacy Metrics":
        show_privacy_metrics(rounds)
    elif page == "Training Metrics":
        show_training_metrics(rounds)
    elif page == "Risk Analysis":
        show_risk_analysis(rounds)
    elif page == "Privacy Ledger":
        show_privacy_ledger(rounds, summary)


def show_overview(rounds: list, summary: dict):
    """Show overview page"""
    st.header("📊 Overview")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rounds", summary.get("total_rounds", 0))
    
    with col2:
        avg_risk = summary.get("statistics", {}).get("risk_scores", {}).get("mean", 0)
        st.metric("Avg Risk Score", f"{avg_risk:.3f}")
    
    with col3:
        avg_noise = summary.get("statistics", {}).get("noise_levels", {}).get("mean", 0)
        st.metric("Avg Noise Level", f"{avg_noise:.3f}")
    
    with col4:
        avg_acc = summary.get("statistics", {}).get("train_accuracies", {}).get("mean", 0)
        st.metric("Avg Accuracy", f"{avg_acc:.3f}")
    
    # Summary dashboard plot
    st.subheader("Summary Dashboard")
    fig = create_summary_dashboard(rounds)
    st.pyplot(fig)
    
    # Recent rounds table
    st.subheader("Recent Rounds")
    if rounds:
        recent_rounds = rounds[-5:]
        round_data = []
        for r in recent_rounds:
            for hospital_id, client_data in r.get("clients", {}).items():
                round_data.append({
                    "Round": r["round"],
                    "Hospital": hospital_id,
                    "Risk Score": client_data.get("risk_score", 0),
                    "Noise Level": client_data.get("noise_level", 0),
                    "Train Acc": client_data.get("train_acc", 0),
                    "Samples": client_data.get("num_samples", 0)
                })
        
        if round_data:
            df = pd.DataFrame(round_data)
            st.dataframe(df, use_container_width=True)


def show_privacy_metrics(rounds: list):
    """Show privacy metrics page"""
    st.header("🔒 Privacy Metrics")
    
    # Risk scores over time
    st.subheader("Risk Scores Over Time")
    fig1 = plot_risk_scores_over_time(rounds)
    st.pyplot(fig1)
    
    # Noise levels over time
    st.subheader("Adaptive Noise Levels Over Time")
    fig2 = plot_noise_levels_over_time(rounds)
    st.pyplot(fig2)
    
    # Risk vs Noise scatter
    st.subheader("Risk Score vs Noise Level")
    fig3 = plot_risk_vs_noise(rounds)
    st.pyplot(fig3)


def show_training_metrics(rounds: list):
    """Show training metrics page"""
    st.header("📈 Training Metrics")
    
    # Accuracy over time
    st.subheader("Training Accuracy Over Time")
    fig1 = plot_accuracy_over_time(rounds)
    st.pyplot(fig1)
    
    # Privacy-utility tradeoff
    st.subheader("Privacy-Utility Tradeoff")
    fig2 = plot_privacy_utility_tradeoff(rounds)
    st.pyplot(fig2)
    
    # Training metrics table
    st.subheader("Training Metrics by Round")
    metrics_data = []
    for r in rounds:
        for hospital_id, client_data in r.get("clients", {}).items():
            metrics_data.append({
                "Round": r["round"],
                "Hospital": hospital_id,
                "Train Acc": client_data.get("train_acc", 0),
                "Train Loss": client_data.get("train_loss", 0),
                "Samples": client_data.get("num_samples", 0)
            })
    
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True)


def show_risk_analysis(rounds: list):
    """Show risk analysis page"""
    st.header("⚠️ Risk Analysis")
    
    # Extract hospital risk scores
    hospitals = set()
    for r in rounds:
        hospitals.update(r.get("clients", {}).keys())
    hospitals = sorted(list(hospitals))
    
    # Risk distribution
    st.subheader("Risk Score Distribution")
    risk_data = []
    for r in rounds:
        for hospital_id in hospitals:
            if hospital_id in r.get("clients", {}):
                risk_data.append({
                    "Round": r["round"],
                    "Hospital": hospital_id,
                    "Risk Score": r["clients"][hospital_id].get("risk_score", 0)
                })
    
    if risk_data:
        df_risk = pd.DataFrame(risk_data)
        
        # Risk statistics by hospital
        st.write("Risk Statistics by Hospital")
        risk_stats = df_risk.groupby("Hospital")["Risk Score"].agg(['mean', 'std', 'min', 'max'])
        st.dataframe(risk_stats)
        
        # Risk histogram
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        for hospital_id in hospitals:
            hospital_risks = df_risk[df_risk["Hospital"] == hospital_id]["Risk Score"]
            ax.hist(hospital_risks, alpha=0.5, label=f'Hospital {hospital_id}', bins=20)
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Risk Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # High risk alerts
    st.subheader("High Risk Alerts")
    high_risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.7, 0.05)
    
    high_risk_events = []
    for r in rounds:
        for hospital_id, client_data in r.get("clients", {}).items():
            risk = client_data.get("risk_score", 0)
            if risk > high_risk_threshold:
                high_risk_events.append({
                    "Round": r["round"],
                    "Hospital": hospital_id,
                    "Risk Score": risk,
                    "Noise Level": client_data.get("noise_level", 0),
                    "Timestamp": r.get("timestamp", "N/A")
                })
    
    if high_risk_events:
        df_alerts = pd.DataFrame(high_risk_events)
        st.dataframe(df_alerts, use_container_width=True)
    else:
        st.info(f"No high risk events above threshold {high_risk_threshold}")


def show_privacy_ledger(rounds: list, summary: dict):
    """Show privacy ledger page"""
    st.header("📋 Privacy Ledger")
    
    # Ledger summary
    st.subheader("Ledger Summary")
    st.json(summary)
    
    # Full ledger data
    st.subheader("Full Ledger Data")
    
    # Option to view as JSON or table
    view_mode = st.radio("View Mode", ["Table", "JSON"], horizontal=True)
    
    if view_mode == "Table":
        # Flatten rounds data for table view
        ledger_data = []
        for r in rounds:
            for hospital_id, client_data in r.get("clients", {}).items():
                ledger_data.append({
                    "Round": r["round"],
                    "Timestamp": r.get("timestamp", "N/A"),
                    "Hospital": hospital_id,
                    "Risk Score": client_data.get("risk_score", 0),
                    "Noise Level": client_data.get("noise_level", 0),
                    "Train Acc": client_data.get("train_acc", 0),
                    "Samples": client_data.get("num_samples", 0)
                })
        
        if ledger_data:
            df = pd.DataFrame(ledger_data)
            st.dataframe(df, use_container_width=True)
    else:
        # JSON view
        st.json(rounds)
    
    # Download ledger
    st.subheader("Download Ledger")
    if st.button("Download Privacy Ledger JSON"):
        ledger_json = json.dumps({
            "summary": summary,
            "rounds": rounds
        }, indent=2)
        st.download_button(
            label="Download",
            data=ledger_json,
            file_name="privacy_ledger.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()

