import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

# --- Page Configuration & Theming ---
st.set_page_config(
    layout="wide",
    page_title="Gait Analysis Dashboard"
)

# Custom CSS to set background and ensure text is readable
st.markdown("""
    <style>
    /* Main background color for the app */
    [data-testid="stAppViewContainer"] {
        background-color: #F0F2F6;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
    }

    /* --- TEXT COLOR OVERRIDE --- */
    /* This is the key fix: Force all text to be a dark color */
    body, h1, h2, h3, h4, h5, h6, p, li, label, .st-emotion-cache-16txtl3 {
        color: #262730 !important; /* A standard dark grey for text */
    }

    /* You can optionally style the main title differently */
    [data-testid="stTitle"] {
        color: #1a5276 !important; /* A deep blue for the main title */
    }
    </style>
    """, unsafe_allow_html=True)

# --- Configuration & Data Generation ---
VIDEO_1_PATH = "video_1.mp4"
VIDEO_2_PATH = "video_2.mp4"
VIDEO_DURATION_SECONDS = 6.0
FPS = 30

@st.cache_data
def generate_gait_data(duration, num_points):
    t = np.linspace(0, duration, num_points)
    gait_cycle_duration = 1.1
    w = 2 * np.pi / gait_cycle_duration
    right_hip = -18 * np.cos(w * t) + 12
    left_hip = -18 * np.cos(w * t + np.pi) + 12
    right_knee = 35 * (1 - np.cos(w * t + 0.2)) / 2 + 15 * np.sin(w * t - 0.5)**4
    left_knee = 35 * (1 - np.cos(w * t + np.pi + 0.2)) / 2 + 15 * np.sin(w * t + np.pi - 0.5)**4
    right_ankle = 12 * np.sin(w * t - np.pi * 0.45) - 5
    left_ankle = 12 * np.sin(w * t + np.pi - np.pi * 0.45) - 5
    data = pd.DataFrame({
        "Time": t, "Left Hip Flexion": left_hip, "Right Hip Flexion": right_hip,
        "Left Knee Flexion": left_knee, "Right Knee Flexion": right_knee,
        "Left Ankle Dorsiflexion": left_ankle, "Right Ankle Dorsiflexion": right_ankle
    }).set_index("Time")
    return data

def get_ai_insights(t):
    cycle_time = t % 2.2
    if 0 <= cycle_time < 0.2: return {"title": "Right Heel Strike", "finding": "Initiating stance phase on the right leg. Hip is flexed (~25°), knee is near full extension to accept weight.", "status": "Normal"}
    elif 0.2 <= cycle_time < 0.8: return {"title": "Left Swing Phase", "finding": "Left leg is in swing. Peak knee flexion (~65°) and ankle dorsiflexion ensure adequate ground clearance.", "status": "Normal"}
    elif 0.8 <= cycle_time < 1.3: return {"title": "Right Push-Off", "finding": "Powerful ankle plantarflexion detected, propelling the body forward.", "status": "Good"}
    elif 1.3 <= cycle_time < 1.5: return {"title": "Left Heel Strike", "finding": "Symmetry check: Left leg makes initial contact. Angles show good bilateral symmetry.", "status": "Symmetrical"}
    elif 1.5 <= cycle_time < 2.0: return {"title": "Right Swing Phase", "finding": "Right leg is now in swing. Hip flexion is increasing towards its peak.", "status": "Normal"}
    else: return {"title": "Overall Assessment", "finding": "Gait pattern appears stable and rhythmic. Cadence is estimated at ~110 steps/minute.", "status": "Stable"}

def get_frame_at_time(video_path, time_sec):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if ret: return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

# --- Main Application ---
st.title("🏃‍♂️ Real-Time Gait Analysis Dashboard")

if 'play' not in st.session_state: st.session_state.play = False
if 'time' not in st.session_state: st.session_state.time = 0.0

st.sidebar.header("Controls")
if st.sidebar.button("▶️ Play / ⏸️ Pause", use_container_width=True):
    st.session_state.play = not st.session_state.play

if st.sidebar.button("🔄 Reset", use_container_width=True):
    st.session_state.play = False
    st.session_state.time = 0.0
    st.rerun()

time_slider = st.sidebar.slider("Timeline (seconds)", 0.0, VIDEO_DURATION_SECONDS, st.session_state.time, 0.01)
if time_slider != st.session_state.time:
    st.session_state.time = time_slider
    st.session_state.play = False

gait_data = generate_gait_data(VIDEO_DURATION_SECONDS, int(VIDEO_DURATION_SECONDS * FPS))
current_data_index = min(int(st.session_state.time * FPS), len(gait_data) - 1)
current_data_point = gait_data.iloc[current_data_index]

main_cols = st.columns([2, 1], gap="large")

with main_cols[0]:
    vid_cols = st.columns(2)
    frame1 = get_frame_at_time(VIDEO_1_PATH, st.session_state.time)
    frame2 = get_frame_at_time(VIDEO_2_PATH, st.session_state.time)

    with vid_cols[0]:
        st.subheader("Original Video")
        if frame1 is not None: st.image(frame1)
    with vid_cols[1]:
        st.subheader("3D Motion Overlay")
        if frame2 is not None: st.image(frame2)

    st.markdown("---")
    st.subheader("Joint Angle Plots")

    plot_data_until_now = gait_data[gait_data.index <= st.session_state.time]
    fig, axs = plt.subplots(2, 3, figsize=(15, 6))
    fig.tight_layout(pad=4.0)

    plot_titles = [
        "Left Hip Flexion", "Right Hip Flexion", "Left Knee Flexion",
        "Right Knee Flexion", "Left Ankle Dorsiflexion", "Right Ankle Dorsiflexion"
    ]
    
    for i, title in enumerate(plot_titles):
        ax = axs[i // 3, i % 3]
        ax.plot(gait_data.index, gait_data[title], color='gray', linestyle='--', alpha=0.5)
        if not plot_data_until_now.empty:
            ax.plot(plot_data_until_now.index, plot_data_until_now[title], color='dodgerblue', linewidth=2)
            ax.plot(plot_data_until_now.index[-1], plot_data_until_now[title].iloc[-1], 'o', color='red', markersize=8)
        
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (°)")
        ax.set_xlim([0, VIDEO_DURATION_SECONDS])
        ax.grid(True, linestyle=':', alpha=0.6)

    st.pyplot(fig)

with main_cols[1]:
    st.subheader("🤖 AI Gait Insights")
    insights = get_ai_insights(st.session_state.time)
    status_color = {"Normal": "blue", "Symmetrical": "green", "Good": "green", "Stable": "violet"}.get(insights["status"], "gray")
    with st.container(border=True):
        st.info(f"**Phase:** {insights['title']}")
        st.markdown(f"**Finding:** {insights['finding']}")
        st.markdown(f"**Status:** :{status_color}[{insights['status']}]")

    st.markdown("---")
    st.subheader("Current Data Points")
    metric_cols = st.columns(2)
    metric_cols[0].metric("Left Knee Flexion", f"{current_data_point['Left Knee Flexion']:.1f}°")
    metric_cols[1].metric("Right Knee Flexion", f"{current_data_point['Right Knee Flexion']:.1f}°")
    metric_cols[0].metric("Left Hip Flexion", f"{current_data_point['Left Hip Flexion']:.1f}°")
    metric_cols[1].metric("Right Hip Flexion", f"{current_data_point['Right Hip Flexion']:.1f}°")
    metric_cols[0].metric("Left Ankle Angle", f"{current_data_point['Left Ankle Dorsiflexion']:.1f}°")
    metric_cols[1].metric("Right Ankle Angle", f"{current_data_point['Right Ankle Dorsiflexion']:.1f}°")

# --- Playback Loop ---
if st.session_state.play:
    if st.session_state.time >= VIDEO_DURATION_SECONDS:
        st.session_state.play = False
    else:
        time_increment = 1 / FPS
        st.session_state.time += time_increment
        st.session_state.time = min(st.session_state.time, VIDEO_DURATION_SECONDS)

    time.sleep(0.02)
    st.rerun()
