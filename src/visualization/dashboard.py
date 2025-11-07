"""
Streamlit Dashboard for FLAIR-X with Animation Support
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
    create_summary_dashboard,
    create_animated_dashboard,
    create_round_summary_table
)
from config import LEDGER_PATH
import time


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
        ["Overview", "🎬 Presentation Mode", "Privacy Metrics", "Training Metrics", 
         "Risk Analysis", "Privacy Ledger"]
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
    elif page == "🎬 Presentation Mode":
        show_presentation_mode(rounds, summary)
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


def show_presentation_mode(rounds: list, summary: dict):
    """Show presentation mode with animations and highlights"""
    st.header("🎬 Presentation Mode")
    st.markdown("### Perfect for demonstrations and presentations!")
    
    # Section 1: Round Highlights Table
    st.subheader("📊 Simulated Round Highlights")
    
    # Calculate summary metrics per round
    summary_data = []
    for round_info in rounds:
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
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        
        # Display as styled table
        st.dataframe(
            df_summary.style.background_gradient(subset=['Accuracy'], cmap='Greens')
                           .background_gradient(subset=['Avg Risk'], cmap='Reds_r')
                           .background_gradient(subset=['Avg Noise'], cmap='Blues'),
            use_container_width=True,
            height=min(len(summary_data) * 35 + 38, 400)
        )
        
        # Also show as matplotlib table for export
        with st.expander("📥 Download Table as Image"):
            fig_table = create_round_summary_table(rounds)
            st.pyplot(fig_table)
            
            # Download button
            buf = io.BytesIO()
            fig_table.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="Download Table Image",
                data=buf,
                file_name="round_highlights.png",
                mime="image/png"
            )
    
    st.markdown("---")
    
    # Section 2: Time-Lapse Animation
    st.subheader("🎥 Time-Lapse Visualization")
    st.markdown("Watch the training progress unfold round by round!")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        animation_speed = st.slider(
            "Animation Speed (seconds per round)", 
            min_value=0.5, 
            max_value=3.0, 
            value=1.0, 
            step=0.5
        )
    
    with col2:
        auto_play = st.checkbox("Auto-play animation", value=True)
    
    with col3:
        save_animation = st.button("💾 Generate & Download Animation")
    
    # Live animation simulation
    if auto_play:
        st.info("🎬 Playing animation... (refresh page to restart)")
        
        # Create placeholder for dashboard
        dashboard_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        for i in range(len(rounds)):
            current_round = i + 1
            current_data = rounds[:current_round]
            
            # Update metrics
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate current metrics
                if current_data:
                    last_round = current_data[-1]
                    clients = last_round.get("clients", {})
                    
                    avg_risk = np.mean([c.get("risk_score", 0) for c in clients.values()])
                    avg_noise = np.mean([c.get("noise_level", 0) for c in clients.values()])
                    avg_acc = np.mean([c.get("train_acc", 0) for c in clients.values()])
                    
                    with col1:
                        st.metric("Current Round", current_round)
                    with col2:
                        st.metric("Avg Risk", f"{avg_risk:.3f}", 
                                 delta=f"{avg_risk - 0.5:.3f}" if i > 0 else None,
                                 delta_color="inverse")
                    with col3:
                        st.metric("Avg Noise", f"{avg_noise:.3f}",
                                 delta=f"{avg_noise - 0.5:.3f}" if i > 0 else None,
                                 delta_color="inverse")
                    with col4:
                        st.metric("Accuracy", f"{avg_acc:.3f}",
                                 delta=f"{avg_acc - 0.65:.3f}" if i > 0 else None,
                                 delta_color="normal")
            
            # Update dashboard
            with dashboard_placeholder:
                fig = create_summary_dashboard(current_data)
                st.pyplot(fig)
            
            # Wait before next round
            time.sleep(animation_speed)
    
    # Save animation option
    if save_animation:
        with st.spinner("Generating animation... This may take a minute."):
            import tempfile
            import os
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                # Generate animation
                create_animated_dashboard(
                    rounds, 
                    save_path=tmp_path,
                    fps=1,
                    duration_per_round=animation_speed
                )
                
                # Read and offer download
                with open(tmp_path, 'rb') as f:
                    animation_data = f.read()
                
                st.success("✅ Animation generated successfully!")
                st.download_button(
                    label="📥 Download Animation (GIF)",
                    data=animation_data,
                    file_name="flair_x_animation.gif",
                    mime="image/gif"
                )
                
            except Exception as e:
                st.error(f"Error generating animation: {e}")
                st.info("💡 Tip: Make sure you have 'pillow' installed for GIF support.")
            
            finally:
                # Clean up
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    
    st.markdown("---")
    
    # Section 3: Key Insights
    st.subheader("🔑 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📉 Risk Trend")
        if rounds:
            first_round = rounds[0]
            last_round = rounds[-1]
            
            first_risk = np.mean([c.get("risk_score", 0) 
                                 for c in first_round.get("clients", {}).values()])
            last_risk = np.mean([c.get("risk_score", 0) 
                                for c in last_round.get("clients", {}).values()])
            
            risk_improvement = ((first_risk - last_risk) / first_risk) * 100
            
            if risk_improvement > 0:
                st.success(f"✅ Risk decreased by {risk_improvement:.1f}%")
                st.markdown(f"From **{first_risk:.3f}** to **{last_risk:.3f}**")
            else:
                st.warning(f"⚠️ Risk increased by {abs(risk_improvement):.1f}%")
    
    with col2:
        st.markdown("#### 📈 Accuracy Trend")
        if rounds:
            first_acc = np.mean([c.get("train_acc", 0) 
                                for c in first_round.get("clients", {}).values()])
            last_acc = np.mean([c.get("train_acc", 0) 
                               for c in last_round.get("clients", {}).values()])
            
            acc_improvement = ((last_acc - first_acc) / first_acc) * 100
            
            if acc_improvement > 0:
                st.success(f"✅ Accuracy improved by {acc_improvement:.1f}%")
                st.markdown(f"From **{first_acc:.3f}** to **{last_acc:.3f}**")
            else:
                st.error(f"❌ Accuracy decreased by {abs(acc_improvement):.1f}%")


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


# Add missing import
import io


if __name__ == "__main__":
    main()