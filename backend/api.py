from fastapi import FastAPI
from backend.user_state_analyzer import analyze_user_state

app = FastAPI()

latest_data = {
    "heart_rate": None,
    "motion_level": None,
    "gsr_value": None,
    "mood": None,
    "confidence": None,
    "metrics": {}
}

user_state_memory = {
    "hr_ema": None,
    "gsr_ema": None,
    "hr_baseline": 75
}

ALLOWED_MOODS = [
    "high stress intense energetic music",
    "stressed tense fast rhythmic music",
    "excited upbeat high energy music",
    "engaged active focused music",
    "deep relaxed calm ambient music",
    "relaxed soft gentle music",
    "low energy slow soft music",
    "neutral balanced background music",
    "low activation minimal ambient music",
    "no_signal"
]


@app.post("/sensor")
def receive_sensor(data: dict):
    global latest_data

    try:
        state = analyze_user_state(data, user_state_memory)

        raw_mood = state["mood"]
        print("RAW MOOD:", raw_mood)

        mood = raw_mood if raw_mood in ALLOWED_MOODS else "neutral balanced background music"

        latest_data = {
            "heart_rate": data.get("heart_rate"),
            "motion_level": data.get("motion_level"),
            "gsr_value": data.get("gsr_value"),
            "mood": mood,
            "confidence": state.get("confidence"),
            "metrics": state.get("metrics", {})
        }

        return {
            "status": "received",
            "detected_mood": mood,
            "confidence": state.get("confidence")
        }

    except Exception as e:
        print("ERROR:", e)
        return {"status": "error", "message": str(e)}


@app.get("/latest")
def get_latest():
    return latest_data