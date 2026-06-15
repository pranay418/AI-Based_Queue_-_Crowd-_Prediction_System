import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# 1. Set Page Configuration
st.set_page_config(
    page_title="AI Crowd & Queue Prediction System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Premium CSS Styling
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    .stApp {
        background-color: #0b0f19;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #f1f5f9;
    }
    
    /* Header & Titles */
    h1, h2, h3 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }
    
    .main-title {
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphic Card Container */
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }
    
    /* Custom Alert Badges */
    .badge-heavy {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model Loading & Utility Functions
@st.cache_resource
def load_ml_model():
    if os.path.exists('queue_model.pkl'):
        return joblib.load('queue_model.pkl')
    return None

@st.cache_data
def load_historical_data():
    if os.path.exists('crowd_data.csv'):
        return pd.read_csv('crowd_data.csv')
    return None

def load_metrics_metadata():
    if os.path.exists('model_metrics.pkl'):
        return joblib.load('model_metrics.pkl')
    return None

# Load Resources
model = load_ml_model()
df_history = load_historical_data()
metrics_meta = load_metrics_metadata()

# Header layout
st.markdown('<div class="main-title">👥 Crowd Analytics & Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time smart queue management and load forecasting powered by AI.</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🛠️ System Control Panel")
st.sidebar.markdown("Configure options below to run forecasting and manage dashboard actions.")

# Day list and mapping
day_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_number = {day: idx for idx, day in enumerate(day_list)}

# 4. Check if Model exists
if model is None:
    st.warning("⚠️ Predictive Model not found or needs initialization!")
    if st.button("Train Predictive Model now"):
        with st.spinner("Training Random Forest Regressor on crowd_data.csv..."):
            try:
                import subprocess
                result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("🎉 Model trained successfully!")
                    st.rerun()
                else:
                    st.error(f"Training failed: {result.stderr}")
            except Exception as e:
                st.error(f"Error starting training: {e}")
    st.stop()

# 5. Create Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Forecast Predictor", 
    "📊 Historical Insights", 
    "📹 CCTV Simulator", 
    "⚙️ Model Control Center"
])

# ==================== TAB 1: LIVE FORECAST PREDICTOR ====================
with tab1:
    st.markdown("### 🔮 Predict Crowd Load & Wait Times")
    st.markdown("Set parameters to query the Random Forest model for future crowd prediction.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Query Parameters")
        
        # User input controls
        sel_hour = st.slider("Select Hour of Day", 0, 23, 12, help="Hour in 24h format")
        sel_day = st.selectbox("Select Day of Week", day_list, index=0)
        
        # Calculate benchmark wait time
        wait_time_per_person = st.number_input(
            "Service rate (minutes per person)", 
            min_value=0.5, 
            max_value=10.0, 
            value=1.5, 
            step=0.5
        )
        
        predict_btn = st.button("Generate Forecast", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("Forecasting Results")
        
        # Default computation or on click
        num_day = day_number[sel_day]
        prediction = model.predict([[sel_hour, num_day]])
        crowd_est = int(prediction[0])
        
        # Determine status
        if crowd_est < 50:
            status_text = "Low Crowd"
            badge_class = "badge-low"
            color_hex = "#10b981"
            recommendation = "✅ Smooth Operations. Standard staffing levels are sufficient."
        elif crowd_est < 150:
            status_text = "Moderate Crowd"
            badge_class = "badge-moderate"
            color_hex = "#f59e0b"
            recommendation = "⚠️ Elevated Load. Monitor queues; consider opening a backup counter."
        else:
            status_text = "Heavy Crowd"
            badge_class = "badge-heavy"
            color_hex = "#ef4444"
            recommendation = "🚨 High Demand! Deploy emergency counters and redirect queues immediately."
            
        # Display Metrics Cards
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
                <div class="metric-label">Estimated Crowd</div>
                <div class="metric-value" style="color: {color_hex};">{crowd_est}</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 5px;">People</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            # Assume 1 queue counter per 60 people
            active_counters = max(1, crowd_est // 60)
            avg_queue = int(crowd_est / active_counters)
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
                <div class="metric-label">Average Queue Length</div>
                <div class="metric-value">{avg_queue}</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 5px;">Per Counter ({active_counters} open)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            total_wait_min = int(avg_queue * wait_time_per_person)
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
                <div class="metric-label">Est. Wait Time</div>
                <div class="metric-value">{total_wait_min}</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 5px;">Minutes</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Load Level Badge
        st.markdown(f"**Current Status:** <span class='{badge_class}'>{status_text}</span>", unsafe_allow_html=True)
        
        # Capacity Progress Bar
        max_capacity = 300
        load_pct = min(1.0, float(crowd_est) / max_capacity)
        st.progress(load_pct, text=f"Capacity Utilization: {int(load_pct * 100)}%")
        
        # Actionable Recommendation Card
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.02); border-left: 4px solid {color_hex}; padding: 15px; border-radius: 0 12px 12px 0; margin-top: 15px;">
            <div style="font-weight: 600; font-size: 1rem; color: #f1f5f9; margin-bottom: 5px;">📋 Operational Guidelines</div>
            <div style="color: #cbd5e1; font-size: 0.95rem;">{recommendation}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 2: HISTORICAL INSIGHTS ====================
with tab2:
    st.markdown("### 📊 Historical Queue & Crowd Density Analysis")
    st.markdown("Explore distribution and trend analytics from the recorded crowd records.")
    
    if df_history is not None:
        # Layout columns
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Hourly Peak Crowd Levels (Average)")
            
            # Compute average crowd by hour
            hourly_avg = df_history.groupby('hour')['people_count'].mean().reset_index()
            
            # Streamlit bar chart
            st.bar_chart(
                hourly_avg.set_index('hour'),
                color="#60a5fa",
                use_container_width=True
            )
            st.caption("This chart displays the typical crowd density across the standard operating hours of the day.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Day-of-Week Load Variance (Average)")
            
            # Compute average crowd by day of week
            day_avg = df_history.groupby('day_of_week')['people_count'].mean().reset_index()
            # Map numbers to names
            day_avg['day_name'] = day_avg['day_of_week'].map(lambda x: day_list[x] if x < len(day_list) else str(x))
            
            # Streamlit line chart
            st.line_chart(
                day_avg.set_index('day_name')['people_count'],
                color="#a78bfa",
                use_container_width=True
            )
            st.caption("This chart visualizes which days see the highest footfall on average.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Detailed Data Table view
        with st.expander("📂 View Raw Historical Records"):
            st.dataframe(
                df_history, 
                column_config={
                    "hour": st.column_config.NumberColumn("Hour (24h)", format="%d"),
                    "day_of_week": st.column_config.NumberColumn("Day of Week Index", format="%d"),
                    "people_count": st.column_config.NumberColumn("Count of People", format="%d")
                },
                use_container_width=True
            )
    else:
        st.info("No historical data file (`crowd_data.csv`) found.")

# ==================== TAB 3: CCTV SIMULATOR ====================
with tab3:
    st.markdown("### 📹 Live CCTV Video Analytics Simulator")
    st.markdown("Simulate computer vision-based counting in real-time on live camera feeds.")
    
    col_cam, col_metrics = st.columns([3, 2])
    
    with col_cam:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Camera Feed - Zone 1A")
        
        # Load generated CCTV mockup image if exists
        if os.path.exists('cctv_feed.png'):
            image = Image.open('cctv_feed.png')
            st.image(image, use_container_width=True, caption="Live CCTV Stream - Terminal Waiting Area")
        else:
            st.info("CCTV Mockup Image not found. Place a 'cctv_feed.png' in the root directory.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_metrics:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("Live CV Analytics")
        
        # Simulation controls
        run_sim = st.toggle("Start Live Stream Analysis", value=False)
        
        placeholder_count = st.empty()
        placeholder_status = st.empty()
        placeholder_fps = st.empty()
        placeholder_chart = st.empty()
        
        sim_counts = []
        
        if run_sim:
            for i in range(10):
                # Simulate frame analysis
                sim_count = np.random.randint(40, 220)
                sim_counts.append(sim_count)
                
                # Update UI elements
                placeholder_count.markdown(f"🤖 **Detected Count:** `{sim_count}` people in frame")
                
                # Visual Alert
                if sim_count < 80:
                    status_badge = "<span class='badge-low'>LOW LOAD</span>"
                elif sim_count < 160:
                    status_badge = "<span class='badge-moderate'>MODERATE LOAD</span>"
                else:
                    status_badge = "<span class='badge-heavy'>HEAVY LOAD (WARNING)</span>"
                placeholder_status.markdown(f"📢 **Load Alert Level:** {status_badge}", unsafe_allow_html=True)
                
                # Frame rate simulation
                fps = np.random.uniform(24.0, 30.0)
                placeholder_fps.markdown(f"⚡ **Processing Performance:** `{fps:.2f} FPS` (GPU-accelerated)")
                
                # Small live chart
                if len(sim_counts) > 1:
                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    plt.style.use('dark_background')
                    ax.plot(sim_counts, color="#60a5fa", marker='o', linewidth=2)
                    ax.set_title("Live Crowd Count (Last 10 Frames)", fontsize=9, color="#94a3b8")
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(True, linestyle='--', alpha=0.2)
                    fig.patch.set_facecolor('#0f172a')
                    ax.set_facecolor('#0f172a')
                    ax.tick_params(labelsize=8)
                    placeholder_chart.pyplot(fig)
                    plt.close(fig)
                
                time.sleep(1.0)
            
            st.success("Simulation batch finished. Toggle to restart.")
        else:
            placeholder_count.info("Click 'Start Live Stream Analysis' above to simulate real-time AI crowd detection.")
            
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 4: MODEL CONTROL CENTER ====================
with tab4:
    st.markdown("### ⚙️ Machine Learning Model Control Center")
    st.markdown("Monitor training parameters, review model accuracy, and trigger online model retraining.")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Model Diagnostic Metrics")
    
    if metrics_meta is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Training Dataset Size", f"{metrics_meta.get('dataset_size', 'N/A')} records")
        m2.metric("Mean Absolute Error (MAE)", f"{metrics_meta.get('mae', 0.0):.2f} people")
        m3.metric("Root Mean Squared Error (RMSE)", f"{metrics_meta.get('rmse', 0.0):.2f} people")
        # Handle formatting of R2 score nicely
        r2_val = metrics_meta.get('r2', 0.0)
        m4.metric("Model R² Score", f"{r2_val:.4f}")
        
        st.info("ℹ️ R² score indicates how well the model predicts variance. A lower value indicates the demo dataset has high randomness.")
    else:
        st.info("No metrics metadata found. Please retrain the model below to generate metrics.")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Retraining section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Model Training Pipeline")
    st.markdown("Clicking the button below loads `crowd_data.csv`, splits the data 80/20, trains a new Random Forest Regressor model, and saves the new model to `queue_model.pkl`.")
    
    if st.button("🔄 Execute Model Retraining Pipeline", use_container_width=True):
        with st.spinner("Retraining model, please wait..."):
            try:
                import subprocess
                result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("🎉 Model retrained and metrics updated successfully!")
                    st.rerun()
                else:
                    st.error(f"Retraining process failed:\n\n{result.stderr}")
            except Exception as e:
                st.error(f"Failed to execute training script: {e}")
                
    st.markdown('</div>', unsafe_allow_html=True)
