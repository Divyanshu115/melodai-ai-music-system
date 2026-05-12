import math

print("🔥 FINAL ANALYZER LOADED")

# ---------------- UTIL ----------------
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(x, hi))

def ema(current, prev, alpha=0.3):
    if prev is None:
        return current
    return alpha * current + (1 - alpha) * prev


# ---------------- MAIN FUNCTION ----------------
def analyze_user_state(sensor_data: dict, state: dict):

    # -------- Validate keys --------
    for k in ["heart_rate", "gsr_value"]:
        if k not in sensor_data:
            raise ValueError("Missing required sensor fields")

    hr = float(sensor_data["heart_rate"])
    gsr = float(sensor_data["gsr_value"])

    print("HR:", hr, "GSR:", gsr)

    # =========================================================
    # 🔥 STEP 1: INVALID DATA CHECK
    # =========================================================
    if hr <= 0 or hr < 40 or hr > 180 :
        return {
            "mood": "no_signal",
            "confidence": 0.0,
            "metrics": {"reason": "invalid_or_unstable_reading"}
        }

    # =========================================================
    # 🔥 STEP 2: SMOOTH SIGNALS
    # =========================================================
    state["hr_ema"] = ema(hr, state.get("hr_ema"), alpha=0.35)
    state["gsr_ema"] = ema(gsr, state.get("gsr_ema"), alpha=0.25)

    hr_s = state["hr_ema"]
    gsr_s = state["gsr_ema"]

    # =========================================================
    # 🔥 STEP 3: HR BASELINE (ONLY HR)
    # =========================================================
    b_alpha = 0.01
    state["hr_baseline"] = ema(hr_s, state.get("hr_baseline", hr_s), alpha=b_alpha)

    hr_base = state["hr_baseline"]

    # =========================================================
    # 🔥 STEP 4: HR RELATIVE
    # =========================================================
    hr_pct = (hr_s - hr_base) / max(hr_base, 1e-6)

    if abs(hr_pct) < 0.01:
        hr_pct = 0

    hr_score = clamp((hr_pct + 0.05) / 0.20)

    # =========================================================
    # 🔥 STEP 5: GSR (INVERTED LOGIC + YOUR RANGES)
    # =========================================================
    GSR_MIN = 300
    GSR_MAX = 2600

    gsr_norm = clamp((gsr_s - GSR_MIN) / (GSR_MAX - GSR_MIN))

    # 🔥 INVERT (LOW = STRESS)
    gsr_stress = (1.0 - gsr_norm) ** 1.1
    gsr_calm = (gsr_norm) ** 1.1

    # =========================================================
    # 🔥 STEP 6: COMBINED METRICS
    # =========================================================
    activation_level = clamp(0.6 * hr_score + 0.4 * (1 - gsr_norm))

    stress_level = clamp(0.8 * gsr_stress + 0.2 * hr_score)

    calm_level = clamp(0.8 * gsr_calm + 0.2 * (1 - hr_score))

    # =========================================================
    # 🔥 STEP 7: HARD ZONE CORRECTION (CRITICAL)
    # =========================================================
    if 0 <= gsr < 600:
        stress_level = max(stress_level, 0.8)

    elif 600 <= gsr < 1200:
        stress_level = max(stress_level, 0.55)

    elif 1200 <= gsr < 2000:
        stress_level = min(stress_level, 0.4)

    elif gsr >= 2000:
        calm_level = max(calm_level, 0.7)

    # =========================================================
    # 🔥 STEP 8: MOOD CLASSIFICATION
    # =========================================================
    if stress_level > 0.75 and activation_level > 0.6:
        mood = "high stress intense energetic music"

    elif stress_level > 0.6:
        mood = "stressed tense fast rhythmic music"

    elif activation_level > 0.7:
        mood = "excited upbeat high energy music"

    elif 0.45 < activation_level <= 0.7:
        mood = "engaged active focused music"

    elif calm_level > 0.7:
        mood = "deep relaxed calm ambient music"

    elif calm_level > 0.5:
        mood = "relaxed soft gentle music"

    elif activation_level < 0.25 and stress_level < 0.4:
        mood = "low energy slow soft music"

    elif 0.25 < activation_level <= 0.55 and stress_level < 0.55:
        mood = "neutral balanced background music"

    else:
        mood = "low activation minimal ambient music"

    # =========================================================
    # 🔥 STEP 9: CONFIDENCE
    # =========================================================
    magnitude = clamp((abs(hr_pct) + gsr_stress) / 1.2)
    confidence = 0.5 + 0.5 * magnitude

    # Debug
    print("HR_BASE:", round(hr_base, 2))
    print("HR_PCT:", round(hr_pct, 3))
    print("GSR_NORM:", round(gsr_norm, 3))
    print("STRESS:", round(stress_level, 3),
          "ACT:", round(activation_level, 3),
          "CALM:", round(calm_level, 3))

    return {
        "mood": mood,
        "confidence": round(confidence, 2),
        "metrics": {
            "hr_change": round(hr_pct, 3),
            "gsr_norm": round(gsr_norm, 3),
            "activation_level": round(activation_level, 3),
            "stress_level": round(stress_level, 3),
            "calm_level": round(calm_level, 3)
        }
    }