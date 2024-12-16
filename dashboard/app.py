import streamlit as st
import requests
import time
import json
import pandas as pd
import altair as alt
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure Streamlit page
st.set_page_config(
    page_title="Federated Learning Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
    <style>
    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 15px;
        border-radius: 10px;
    }
    .stMetric:hover {
        background-color: rgba(28, 131, 225, 0.2);
    }
    .plot-container {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
    }
    .worker-card {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:3000"

def fetch_status():
    try:
        response = requests.get(f"{API_URL}/status")
        return response.json()
    except requests.exceptions.RequestException:
        return None

def calculate_convergence_rate(history):
    if len(history) < 2:
        return 0
    losses = [h['global_loss'] for h in history]
    if len(losses) < 2:
        return 0
    
    # Calculate rates while avoiding division by zero
    rates = []
    for i in range(len(losses)-1):
        if losses[i] != 0:  # Avoid division by zero
            rate = (losses[i] - losses[i+1])/losses[i]
        else:
            # If current loss is 0, check if next loss is different
            rate = -1 if losses[i+1] > 0 else 0
        rates.append(rate)
    
    return np.mean(rates) if rates else 0

def format_delta(value):
    """Format delta values for metrics, handling edge cases"""
    if value is None or np.isnan(value):
        return None
    return f"{value:.2%}"

def create_loss_chart(history_df):
    fig = go.Figure()
    
    # Add global loss line
    fig.add_trace(go.Scatter(
        x=history_df['epoch'],
        y=history_df['global_loss'],
        name='Global Loss',
        line=dict(color='#1f77b4', width=3),
        mode='lines+markers'
    ))
    
    # Add trend line only if we have enough data points
    if len(history_df) > 2:
        try:
            z = np.polyfit(history_df['epoch'], history_df['global_loss'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=history_df['epoch'],
                y=p(history_df['epoch']),
                name='Trend',
                line=dict(color='rgba(255, 255, 255, 0.5)', dash='dash'),
                mode='lines'
            ))
        except np.linalg.LinAlgError:
            # Skip trend line if fitting fails
            pass
    
    fig.update_layout(
        title='Training Progress',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        template='plotly_dark',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_worker_comparison(status):
    if not status or not status['worker_losses']:
        return None
        
    worker_data = pd.DataFrame([
        {'worker': w, 'loss': l} 
        for w, l in status['worker_losses'].items()
    ])
    
    fig = px.bar(
        worker_data,
        x='worker',
        y='loss',
        color='loss',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        title='Worker Performance Comparison',
        template='plotly_dark',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Initialize session state
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'convergence_history' not in st.session_state:
    st.session_state.convergence_history = []

# Title and description
st.title("🤖 Federated Learning Dashboard")
st.markdown("""
    Real-time monitoring of federated learning training across distributed workers.
    Track global model convergence and individual worker performance.
""")

# Main dashboard layout
status = fetch_status()

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

if status:
    current_time = datetime.now()
    if (current_time - st.session_state.last_update).seconds >= 2:
        st.session_state.training_history.append({
            'epoch': status['current_epoch'],
            'global_loss': status['global_loss'] if status['global_loss'] else 0.0,
            'timestamp': current_time
        })
        
        # Calculate convergence rate
        conv_rate = calculate_convergence_rate(st.session_state.training_history)
        st.session_state.convergence_history.append({
            'epoch': status['current_epoch'],
            'rate': conv_rate
        })
        
        st.session_state.last_update = current_time

    with col1:
        st.metric(
            "Current Epoch",
            status['current_epoch'],
            delta=None,
            help="Current training epoch"
        )

    with col2:
        if status['global_loss'] is not None:
            conv_rate = calculate_convergence_rate(st.session_state.training_history)
            delta = format_delta(-conv_rate) if conv_rate != 0 else None
            st.metric(
                "Global Loss",
                f"{status['global_loss']:.6f}",
                delta=delta,
                delta_color="inverse",
                help="Current global model loss"
            )

    with col3:
        st.metric(
            "Active Workers",
            len(status['active_workers']),
            help="Number of workers currently participating"
        )

    with col4:
        total_samples = sum(status.get('worker_samples', {}).values())
        st.metric(
            "Total Training Samples",
            f"{total_samples:,}",
            help="Total number of training samples across all workers"
        )

    # Main charts row
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📈 Training Progress")
        if st.session_state.training_history:
            df = pd.DataFrame(st.session_state.training_history)
            loss_chart = create_loss_chart(df)
            st.plotly_chart(loss_chart, use_container_width=True)

    with col2:
        st.markdown("### 👥 Worker Performance")
        worker_chart = create_worker_comparison(status)
        if worker_chart:
            st.plotly_chart(worker_chart, use_container_width=True)

    # Worker details section
    st.markdown("### 🔍 Worker Details")
    worker_cols = st.columns(3)
    for idx, worker in enumerate(status['active_workers']):
        with worker_cols[idx % 3]:
            worker_loss = status['worker_losses'].get(worker, 0.0)
            st.markdown(f"""
                <div class="worker-card">
                    <h4>{worker}</h4>
                    <p>Current Loss: {worker_loss:.6f}</p>
                    <p>Training Samples: {status.get('worker_samples', {}).get(worker, 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)

    # Training insights
    if len(st.session_state.training_history) > 1:
        st.markdown("### 🎯 Training Insights")
        insight_cols = st.columns(2)
        
        with insight_cols[0]:
            convergence_rate = calculate_convergence_rate(st.session_state.training_history)
            stability = (
                "Stable" if convergence_rate > 0 
                else "Unstable" if convergence_rate < 0 
                else "Initializing"
            )
            st.markdown(f"""
                📊 **Convergence Analysis**
                - Current convergence rate: {format_delta(convergence_rate) or 'Calculating...'}
                - Training stability: {stability}
            """)
        
        with insight_cols[1]:
            worker_losses = status.get('worker_losses', {})
            if worker_losses:
                best_worker = min(worker_losses.items(), key=lambda x: x[1])[0]
                best_worker_text = f"Best performing worker: {best_worker}"
            else:
                best_worker_text = "Waiting for worker updates..."
            
            st.markdown(f"""
                🏆 **Performance Highlights**
                - {best_worker_text}
                - Global model updates: {status['current_epoch'] * len(status['active_workers'])}
            """)

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    auto_refresh = st.checkbox("Auto Refresh", value=True)
    refresh_interval = st.slider("Refresh Interval (seconds)", 1, 10, 2)
    
    if st.button("Force Refresh"):
        st.rerun()

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
