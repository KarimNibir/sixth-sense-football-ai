"""
Narrative Engine — generates tactical insights using Groq (free tier).
Two modes: ANALYST (deep tactical) and FAN (plain English, accessible).
Designed to swap to Azure OpenAI with one env variable change.
"""
import os, json
from groq import Groq

# ── Model config ─────────────────────────────────────────────────────────────
# Groq free tier. To switch to Azure OpenAI:
#   Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY instead of GROQ_API_KEY
GROQ_MODEL   = "qwen/qwen3.8-27b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

ANALYST_SYSTEM = """You are an elite football tactical analyst providing real-time match intelligence.
Given current match data, generate a 2-3 sentence insight that:
1. Names the specific tactical pattern you see developing
2. Explains WHY this moment is significant (not just what happened)
3. Predicts what to watch for in the next 3-5 minutes

Be precise. Use tactical vocabulary. Cite the numbers. Sound like a UEFA Pro Licence coach, not a commentator."""

FAN_SYSTEM = """You are an enthusiastic football expert explaining the match to a casual fan.
Given current match data, generate 2 sentences that:
1. Explain what's happening in plain English (no jargon)
2. Tell them what exciting thing might happen next

Be energetic, clear, and make the fan feel like they understand what's going on. Use simple language."""

def _groq_client():
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    return Groq(api_key=key)

def _build_prompt(tension_data: dict, mode: str) -> str:
    stats    = tension_data.get("stats", {})
    patterns = tension_data.get("patterns", [])
    tension  = tension_data.get("tension", 0)

    pattern_names = ", ".join(p["name"] for p in patterns) if patterns else "No clear pattern yet"
    pattern_descs = "; ".join(p["description"] for p in patterns) if patterns else "Play is neutral"

    return f"""MATCH STATE (minute {stats.get('minute', '?')}):
- Tension Score: {tension}/100
- Detected Patterns: {pattern_names}
- Pattern Details: {pattern_descs}
- Passes in last 15 events: {stats.get('passes', 0)} ({stats.get('pass_accuracy', 0)}% accuracy)
- Shots: {stats.get('shots', 0)}
- Pressures: {stats.get('pressures', 0)}
- Play in danger zone: {stats.get('in_danger_zone_pct', 0)}%
- Forward pass rate: {stats.get('forward_pass_rate', 0)}%
- Teams in sequence: {', '.join(stats.get('teams_in_window', []))}

Generate a {"tactical analyst" if mode=="analyst" else "fan-friendly"} insight."""

def narrate(tension_data: dict, mode: str = "analyst") -> str:
    """
    Generate a narrative insight.
    mode: "analyst" or "fan"
    Returns a string.
    """
    client  = _groq_client()
    system  = ANALYST_SYSTEM if mode == "analyst" else FAN_SYSTEM
    prompt  = _build_prompt(tension_data, mode)

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=150,
            temperature=0.6
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Fallback to smaller model
        try:
            resp = client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=150,
                temperature=0.6
            )
            return resp.choices[0].message.content.strip()
        except:
            return _fallback_narrative(tension_data, mode)

def _fallback_narrative(tension_data: dict, mode: str) -> str:
    tension  = tension_data.get("tension", 0)
    patterns = tension_data.get("patterns", [])
    stats    = tension_data.get("stats", {})
    minute   = stats.get("minute", 0)

    if mode == "fan":
        if tension > 70:
            return f"⚡ Something big is building at minute {minute}! The attacking team is pushing hard and a goal could come any moment."
        elif tension > 40:
            return f"👀 Interesting phase of play at minute {minute}. The game is opening up — keep watching."
        else:
            return f"⚽ Quiet spell at minute {minute}. Both teams are finding their shape."
    else:
        top = patterns[0]["description"] if patterns else "Neutral possession phase"
        return f"Minute {minute}: Tension index {tension}/100. {top}. Monitor transition speed and defensive line depth."

def narrate_both(tension_data: dict) -> dict:
    """Returns both analyst and fan narratives."""
    return {
        "analyst": narrate(tension_data, "analyst"),
        "fan":     narrate(tension_data, "fan")
    }
