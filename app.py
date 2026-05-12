import streamlit as st
import os
import time
import requests

from backend.main_service import generate_music_pipeline
from backend.quality_scorer import QualityScorer, default_meta_to_audio_path

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="MelodAI – AI Music Generator",
    page_icon="🎵",
    layout="wide"
)

# ------------------------------------------------------
# TITLE
# ------------------------------------------------------
st.title("🎶 MelodAI – AI Music Generator")
st.caption("Adaptive AI Music Generation using Live Sensor Data")

# ------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------
if "last_meta" not in st.session_state:
    st.session_state.last_meta = None

if "sensor_data" not in st.session_state:
    st.session_state.sensor_data = None

if "detected_prompt" not in st.session_state:
    st.session_state.detected_prompt = ""

# ------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------
st.sidebar.header("⚙️ Controls")

duration = st.sidebar.slider("Duration (seconds)", 5, 60, 15)
variations = st.sidebar.slider("Variations", 1, 5, 1)
retries = st.sidebar.slider("Retries", 1, 5, 2)

model_choice = st.sidebar.selectbox(
    "Model Preference",
    ["auto", "facebook/musicgen-small", "facebook/musicgen-medium", "facebook/musicgen-large"]
)

quality_pref = st.sidebar.selectbox(
    "Quality Preference",
    ["balanced", "fast", "quality"]
)

debug = st.sidebar.checkbox("Debug mode")

# ------------------------------------------------------
# SENSOR SECTION (LIVE FROM BACKEND)
# ------------------------------------------------------
st.subheader("📡 Live Sensor Data")

if st.button("🔄 Fetch Latest Sensor Data"):
    try:
        response = requests.get("http://127.0.0.1:8000/latest")
        data = response.json()

        st.session_state.sensor_data = data
        st.session_state.detected_prompt = f"{data['mood']}  music"

        st.success("Sensor data fetched successfully")

    except:
        st.error("Backend not reachable")

# Display Sensor Data (Read-only)
if st.session_state.sensor_data:

    data = st.session_state.sensor_data

    col1, col2, col3 = st.columns(3)

    col1.metric("Heart Rate", data["heart_rate"])
    col2.metric("Motion Level", data["motion_level"])
    col3.metric("GSR", data["gsr_value"])

    st.success(f"Detected Mood: {data['mood']}")

# ------------------------------------------------------
# PROMPT INPUT (AUTO-FILLED FROM BACKEND)
# ------------------------------------------------------
prompt_text = st.text_area(
    "🎧 Generated Mood Prompt",
    value=st.session_state.detected_prompt,
    height=120
)

col1, col2 = st.columns([1, 1])

# ------------------------------------------------------
# GENERATE BUTTON
# ------------------------------------------------------
with col1:
    if st.button("🚀 Generate Music", use_container_width=True):

        final_prompt = prompt_text.strip()

        if not final_prompt:
            st.warning("Fetch sensor data first.")
        else:
            with st.spinner("Generating music... 🎼"):
                meta, params, enhanced_prompts = generate_music_pipeline(
                    user_input=final_prompt,
                    duration=duration,
                    variations=variations,
                    retries=retries,
                    debug=debug,
                    model_name=None if model_choice == "auto" else model_choice,
                    quality_preference=quality_pref,
                )

                st.session_state.last_meta = meta

            st.success("Music generated successfully!")

# ------------------------------------------------------
# OUTPUT SECTION
# ------------------------------------------------------
if st.session_state.last_meta:
    meta = st.session_state.last_meta

    st.divider()
    st.subheader("🎼 Generated Output")

    audio_path = default_meta_to_audio_path(meta)

    if audio_path and os.path.exists(audio_path):
        st.audio(audio_path)
        st.download_button(
            "⬇️ Download Audio",
            data=open(audio_path, "rb"),
            file_name=os.path.basename(audio_path),
            mime="audio/wav"
        )
    else:
        st.error("Audio file not found.")

    with st.expander("📊 Generation Details"):
        st.json(meta)

    st.subheader("⭐ Quality Score")

    scorer = QualityScorer()
    score_report = scorer.score_audio(
        audio_path,
        expected_params={
            "duration": duration,
            "mood": meta.get("pipeline", {}).get("params", {}).get("mood"),
            "energy": meta.get("pipeline", {}).get("params", {}).get("energy"),
        }
    )

    st.metric("Overall Score", round(score_report.get("overall_score", 0), 2))
    st.json(score_report.get("scores", {}))
