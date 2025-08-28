import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Gait Analysis Dashboard",
    page_icon="🏃‍♂️"
)

# --- Custom CSS for Theming ---
CUSTOM_CSS = """
/* General Theme */
body {
    font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
}

/* Main background */
[data-testid="stAppViewContainer"] {
    background-color: #F0F2F6;
}

/* Card styles */
[data-testid="stVerticalBlock"] .st-emotion-cache-16txtl3 {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.04);
}

/* Sidebar style */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
}

/* Title and Headers */
h1, h2, h3 {
    color: #1a5276; 
}

/* Custom boxes for summary section */
.summary-box {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    color: #333;
}
.normal-box {
    background-color: #e8f5e9; /* Light green */
    border-left: 5px solid #66bb6a;
}
.monitor-box {
    background-color: #fff3e0; /* Light yellow */
    border-left: 5px solid #ffa726;
}
.rec-box {
    background-color: #e3f2fd; /* Light blue */
    border-left: 5px solid #42a5f5;
}
"""
st.markdown(f'<style>{CUSTOM_CSS}</style>', unsafe_allow_html=True)


# --- Configuration & Data Generation (same as before) ---
VIDEO_1_PATH = "video_1.mp4"
VIDEO_2_PATH = "video_2.mp4"
VIDEO_DURATION_SECONDS = 6.0
FPS = 30

@st.cache_data
def generate_gait_data(duration, num_points):
    """Generates synthetic gait data for plotting."""
    t = np.linspace(0, duration, num_points)
    gait_cycle_duration = 1.1
    w = 2 * np.pi / gait_cycle_duration
    right_hip = -18 * np.cos(w * t) + 12
    left_hip = -18 * np.cos(w * t + np.pi) + 12
    right_knee = 35 * (1 - np.cos(w * t + 0.2)) / 2 + 15 * np.sin(w * t - 0.5)**4 + 5
    left_knee = 35 * (1 - np.cos(w * t + np.pi + 0.2)) / 2 + 15 * np.sin(w * t + np.pi - 0.5)**4 + 5
    right_ankle = 12 * np.sin(w * t - np.pi * 0.45) - 5
    left_ankle = 12 * np.sin(w * t + np.pi - np.pi * 0.45) - 5
    data = pd.DataFrame({
        "Time": t, "Left Hip Flexion": left_hip, "Right Hip Flexion": right_hip,
        "Left Knee Flexion": left_knee, "Right Knee Flexion": right_knee,
        "Left Ankle Dorsiflexion": left_ankle, "Right Ankle Dorsiflexion": right_ankle
    }).set_index("Time")
    return data

def get_frame_at_time(video_path, time_sec):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if ret: return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

# --- Main Application ---
st.title("Gait Analysis Dashboard")

# --- Initialize Session State ---
if 'play' not in st.session_state: st.session_state.play = False
if 'time' not in st.session_state: st.session_state.time = 0.0

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Controls")
    if st.button("▶️ Play / ⏸️ Pause", use_container_width=True):
        st.session_state.play = not st.session_state.play

    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.play = False
        st.session_state.time = 0.0
        st.rerun()

    time_slider = st.slider("Timeline (seconds)", 0.0, VIDEO_DURATION_SECONDS, st.session_state.time, 0.01)
    if time_slider != st.session_state.time:
        st.session_state.time = time_slider
        st.session_state.play = False
    
    st.subheader("Current Data")
    gait_data = generate_gait_data(VIDEO_DURATION_SECONDS, int(VIDEO_DURATION_SECONDS * FPS))
    current_data_index = min(int(st.session_state.time * FPS), len(gait_data) - 1)
    current_data_point = gait_data.iloc[current_data_index]

    metric_cols = st.columns(2)
    metric_cols[0].metric("L. Knee", f"{current_data_point['Left Knee Flexion']:.1f}°")
    metric_cols[1].metric("R. Knee", f"{current_data_point['Right Knee Flexion']:.1f}°")
    metric_cols[0].metric("L. Hip", f"{current_data_point['Left Hip Flexion']:.1f}°")
    metric_cols[1].metric("R. Hip", f"{current_data_point['Right Hip Flexion']:.1f}°")
    metric_cols[0].metric("L. Ankle", f"{current_data_point['Left Ankle Dorsiflexion']:.1f}°")
    metric_cols[1].metric("R. Ankle", f"{current_data_point['Right Ankle Dorsiflexion']:.1f}°")


# --- Main Layout ---
top_cols = st.columns(2, gap="large")

with top_cols[0]:
    with st.container(border=True):
        st.subheader("⚡ 3D Mesh Overlay")
        st.caption("Real-time joint tracking and movement analysis")
        
        frame = get_frame_at_time(VIDEO_2_PATH, st.session_state.time)
        if frame is not None:
            st.image(frame)
        
        status_cols = st.columns(2)
        status_cols[0].metric("Hip Joints", "Normal", delta="Stable")
        status_cols[1].metric("Knee Joints", "Normal", delta="Symmetrical")
        status_cols[0].metric("Ankle Joints", "Mild Asymmetry", delta="-1.2° Diff", delta_color="inverse")
        status_cols[1].metric("Spine", "Stable", delta="Good")


with top_cols[1]:
    with st.container(border=True):
        st.subheader("📈 Joint Angle Analysis")
        st.caption("Real-time joint movement patterns and clinical indicators")
        
        plot_titles = [
            "Left Hip Flexion", "Right Hip Flexion", "Left Knee Flexion",
            "Right Knee Flexion", "Left Ankle Dorsiflexion", "Right Ankle Dorsiflexion"
        ]
        selected_joint = st.selectbox("Select Joint to Analyze", plot_titles, index=2)
        
        plot_data_until_now = gait_data[gait_data.index <= st.session_state.time]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot styling to match the image
        full_data = gait_data[selected_joint]
        ax.plot(full_data.index, full_data, color='#28A745', alpha=0.2, linewidth=1)
        ax.fill_between(full_data.index, full_data, color='#D4EDDA', alpha=0.5)

        if not plot_data_until_now.empty:
            partial_data = plot_data_until_now[selected_joint]
            ax.plot(partial_data.index, partial_data, color='#28A745', linewidth=2.5)
            ax.plot(partial_data.index[-1], partial_data.iloc[-1], 'o', color='#E74C3C', markersize=8)

        # Aesthetics
        ax.set_title(f"{selected_joint} Pattern", fontsize=14)
        ax.set_ylabel("Angle (degree)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(min(gait_data[selected_joint]) - 10, max(gait_data[selected_joint]) + 10)
        ax.set_xlim(0, VIDEO_DURATION_SECONDS)
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        st.pyplot(fig)
        
        st.info(f"**Clinical Interpretation:** Excellent {selected_joint.split(' ')[1]} range of motion. Consistent pattern indicates functional stride mechanics.", icon="💡")


# --- Clinical Summary & Recommendations ---
with st.container(border=True):
    st.subheader("ℹ️ Clinical Summary & Recommendations")
    summary_cols = st.columns(3, gap="medium")
    
    with summary_cols[0]:
        st.markdown("""
        <div class="summary-box normal-box">
            <strong>✅ Normal Patterns</strong>
            <ul>
                <li>Hip flexion-extension rhythm</li>
                <li>Knee range of motion (65°)</li>
                <li>Pelvic stability</li>
                <li>Spinal coordination</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with summary_cols[1]:
        st.markdown("""
        <div class="summary-box monitor-box">
            <strong>⚠️ Monitor</strong>
            <ul>
                <li>Ankle variability</li>
                <li>Slight asymmetry in rotation</li>
                <li>Gait cycle consistency</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with summary_cols[2]:
        st.markdown("""
        <div class="summary-box rec-box">
            <strong>🔬 Recommendations</strong>
            <ul>
                <li>Continue monitoring</li>
                <li>Balance assessment</li>
                <li>Follow-up in 6 months</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# --- Playback Loop Logic ---
if st.session_state.play:
    if st.session_state.time >= VIDEO_DURATION_SECONDS:
        st.session_state.play = False
    else:
        time_increment = 1 / FPS
        st.session_state.time += time_increment
        st.session_state.time = min(st.session_state.time, VIDEO_DURATION_SECONDS)

    time.sleep(0.02)
    st.rerun()
