import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

# --- Configuration ---
VIDEO_1_PATH = "video_1.mp4"
VIDEO_2_PATH = "video_2.mp4"
VIDEO_DURATION_SECONDS = 6  # Manually set the duration of the relevant action
FPS = 30 # Frames per second for data generation and playback simulation

# --- Data Generation ---
# This function creates realistic-looking synthetic data for a walking gait cycle.
@st.cache_data
def generate_gait_data(duration, num_points):
    """Generates synthetic gait data for plotting."""
    t = np.linspace(0, duration, num_points)
    gait_cycle_duration = 1.1  # seconds per full gait cycle
    w = 2 * np.pi / gait_cycle_duration

    # Hip Flexion (positive = flexion)
    right_hip = -18 * np.cos(w * t) + 12
    left_hip = -18 * np.cos(w * t + np.pi) + 12

    # Knee Flexion (always positive)
    # A more complex curve with a major peak in swing and a smaller one in stance
    right_knee = 35 * (1 - np.cos(w * t + 0.2)) / 2 + 15 * np.sin(w * t - 0.5)**4
    left_knee = 35 * (1 - np.cos(w * t + np.pi + 0.2)) / 2 + 15 * np.sin(w * t + np.pi - 0.5)**4

    # Ankle Dorsiflexion (positive = dorsiflexion, negative = plantarflexion)
    right_ankle = 12 * np.sin(w * t - np.pi * 0.45) - 5
    left_ankle = 12 * np.sin(w * t + np.pi - np.pi * 0.45) - 5
    
    data = pd.DataFrame({
        "Time": t,
        "Left Hip Flexion": left_hip,
        "Right Hip Flexion": right_hip,
        "Left Knee Flexion": left_knee,
        "Right Knee Flexion": right_knee,
        "Left Ankle Dorsiflexion": left_ankle,
        "Right Ankle Dorsiflexion": right_ankle
    }).set_index("Time")
    
    return data

# --- AI Insights ---
# This function provides contextual text based on the video's timestamp.
def get_ai_insights(t):
    """Returns AI-driven text insights based on the current time."""
    cycle_time = t % 2.2 # Assume a full L-R cycle is ~2.2 seconds
    if 0 <= cycle_time < 0.2:
        return {
            "title": "Right Heel Strike",
            "finding": "Initiating stance phase on the right leg. Hip is flexed (~25°), knee is near full extension to accept weight.",
            "status": "Normal"
        }
    elif 0.2 <= cycle_time < 0.8:
        return {
            "title": "Left Swing Phase",
            "finding": "Left leg is in swing. Peak knee flexion (~65°) and ankle dorsiflexion ensure adequate ground clearance.",
            "status": "Normal"
        }
    elif 0.8 <= cycle_time < 1.3:
        return {
            "title": "Right Push-Off",
            "finding": "Powerful ankle plantarflexion detected on the right side, propelling the body forward. This is a key indicator of propulsive force.",
            "status": "Good"
        }
    elif 1.3 <= cycle_time < 1.5:
        return {
            "title": "Left Heel Strike",
            "finding": "Symmetry check: Left leg makes initial contact. Comparing angles to the right side shows good bilateral symmetry.",
            "status": "Symmetrical"
        }
    elif 1.5 <= cycle_time < 2.0:
        return {
            "title": "Right Swing Phase",
            "finding": "Right leg is now in swing. Hip flexion is increasing towards its peak.",
            "status": "Normal"
        }
    else:
        return {
            "title": "Overall Assessment",
            "finding": "Gait pattern appears stable and rhythmic. Cadence is estimated at ~110 steps/minute.",
            "status": "Stable"
        }

# --- Video Frame Extraction ---
# Using st.cache_data is crucial for performance on the cloud
@st.cache_data
def get_video_frames(video_path):
    """Extracts all frames from a video and returns them as a list."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

# --- Main Application ---
st.set_page_config(layout="wide", page_title="Gait Analysis Dashboard")

# Check if video files exist (helpful for local debugging)
if not os.path.exists(VIDEO_1_PATH) or not os.path.exists(VIDEO_2_PATH):
    st.error(f"Video files not found! Please make sure '{VIDEO_1_PATH}' and '{VIDEO_2_PATH}' are in the same directory as the script.")
    st.stop()

st.title("🏃‍♂️ Real-Time Gait Analysis Dashboard")

# --- Initialize Session State ---
if 'play' not in st.session_state:
    st.session_state.play = False
if 'time' not in st.session_state:
    st.session_state.time = 0.0

# --- Pre-load video frames ---
frames1 = get_video_frames(VIDEO_1_PATH)
frames2 = get_video_frames(VIDEO_2_PATH)
num_frames = min(len(frames1), len(frames2))
actual_fps = num_frames / VIDEO_DURATION_SECONDS

# --- Sidebar Controls ---
st.sidebar.header("Controls")
play_button = st.sidebar.button("▶️ Play" if not st.session_state.play else "⏸️ Pause")
if play_button:
    st.session_state.play = not st.session_state.play

if st.sidebar.button("🔄 Reset"):
    st.session_state.play = False
    st.session_state.time = 0.0
    st.experimental_rerun()

time_slider = st.sidebar.slider("Timeline (seconds)", 0.0, VIDEO_DURATION_SECONDS, st.session_state.time, 0.01)
if time_slider != st.session_state.time:
    st.session_state.time = time_slider
    st.session_state.play = False

# --- Data Loading ---
gait_data = generate_gait_data(VIDEO_DURATION_SECONDS, int(VIDEO_DURATION_SECONDS * FPS))
current_data_index = int(st.session_state.time * (len(gait_data) / VIDEO_DURATION_SECONDS))
current_data_index = min(current_data_index, len(gait_data) - 1)
current_data_point = gait_data.iloc[current_data_index]

# --- Main Layout ---
main_cols = st.columns([2, 1])

with main_cols[0]:
    vid_cols = st.columns(2)
    frame_index = int(st.session_state.time * actual_fps)
    frame_index = min(frame_index, num_frames - 1)
    
    with vid_cols[0]:
        st.subheader("Original Video")
        st.image(frames1[frame_index])
    with vid_cols[1]:
        st.subheader("3D Motion Overlay")
        st.image(frames2[frame_index])

    st.markdown("---")
    st.subheader("Joint Angle Plots")

    plot_data_until_now = gait_data[gait_data.index <= st.session_state.time]
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 6))
    fig.tight_layout(pad=4.0)

    plot_titles = [
        "Left Hip Flexion", "Right Hip Flexion",
        "Left Knee Flexion", "Right Knee Flexion",
        "Left Ankle Dorsiflexion", "Right Ankle Dorsiflexion"
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
    time_increment = 1 / FPS
    if st.session_state.time >= VIDEO_DURATION_SECONDS:
        st.session_state.time = 0.0
    else:
        st.session_state.time += time_increment
    
    if st.session_state.time > VIDEO_DURATION_SECONDS:
        st.session_state.time = VIDEO_DURATION_SECONDS
        st.session_state.play = False

    time.sleep(0.01) # Small sleep to allow browser to render
    st.experimental_rerun()