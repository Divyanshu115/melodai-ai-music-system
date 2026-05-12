# examples/compare_prompts.py
import json
from pathlib import Path
from backend.prompt_enhancer import PromptEnhancer

# sample parsed outputs from your InputProcessor (10 test cases)
TEST_PARAMS = [
    {"mood":"energetic","energy":9,"style":"electronic dance","tempo":"fast","instruments":["drums","synth","bass"],"context":"workout","raw_text":"I need energetic music for my workout"},
    {"mood":"calm","energy":2,"style":"ambient","tempo":"slow","instruments":["synth pads","flute","soft piano"],"context":"meditation","raw_text":"Something calming for meditation"},
    {"mood":"happy","energy":8,"style":"pop","tempo":"fast","instruments":["drums","synth","bass","guitar"],"context":"birthday party","raw_text":"Happy birthday party music"},
    {"mood":"sad","energy":3,"style":"acoustic ballad","tempo":"slow","instruments":["acoustic guitar","piano","strings"],"context":"breakup","raw_text":"Sad breakup song"},
    {"mood":"calm","energy":4,"style":"ambient","tempo":"slow","instruments":["piano","synth","soft pads"],"context":"study","raw_text":"Focus music for studying"},
    {"mood":"epic","energy":9,"style":"orchestral","tempo":"fast","instruments":["strings","brass","percussion","choir"],"context":"battle scene","raw_text":"Epic cinematic battle scene"},
    {"mood":"romantic","energy":3,"style":"soft romantic","tempo":"slow","instruments":["piano","strings","acoustic guitar"],"context":"background","raw_text":"Soft romantic background track"},
    {"mood":"mysterious","energy":4,"style":"ambient horror","tempo":"slow","instruments":["synth","strings","ambient pads","low drones"],"context":"horror game","raw_text":"Dark mysterious atmosphere for a horror game"},
    {"mood":"calm","energy":3,"style":"lofi hip hop","tempo":"slow","instruments":["piano","drums","bass","synth"],"context":"study","raw_text":"Chill lofi beats to relax and study"},
    {"mood":"energetic","energy":9,"style":"techno","tempo":"fast","instruments":["synth","drums","bass"],"context":"running","raw_text":"Fast techno track for running"}
]

def main():
    # path to mood templates (optional)
    mood_json = Path("backend/mood_templates.json")
    enhancer = PromptEnhancer(mood_json_path=str(mood_json), seed=42)

    out = []
    for idx, p in enumerate(TEST_PARAMS, start=1):
        basic = p.get("raw_text", "")
        enhanced_list = enhancer.enhance(p, variations=2)  # produce 2 variations
        validations = [enhancer.validate_prompt(e) for e in enhanced_list]

        print("="*60)
        print(f"[{idx}] BASIC: {basic}")
        for v_idx, (enhanced, val) in enumerate(zip(enhanced_list, validations), start=1):
            print(f"\n  Enhanced #{v_idx}:\n    {enhanced}\n    VALID: {val}\n")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
