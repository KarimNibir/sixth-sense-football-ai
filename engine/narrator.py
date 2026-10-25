"""
Narrator — dual-mode AI narrative engine
Primary: Azure AI (GitHub Models) via models.inference.ai.azure.com
Fallback: Groq API (OpenAI-compatible)
"""
import os
from openai import OpenAI

# ── Azure AI (GitHub Models) — Microsoft's infrastructure, gpt-4o-mini
AZURE_AI_ENDPOINT = "https://models.inference.ai.azure.com"
AZURE_AI_MODEL    = "gpt-4o-mini"
GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN", "")

# ── Groq fallback
GROQ_ENDPOINT     = "https://api.groq.com/openai/v1"
GROQ_MODEL        = "qwen/qwen3.8-27b"
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")


def _get_client():
    """Return Azure AI client if token available, else Groq."""
    if GITHUB_TOKEN:
        return OpenAI(
            base_url=AZURE_AI_ENDPOINT,
            api_key=GITHUB_TOKEN,
        ), AZURE_AI_MODEL
    return OpenAI(
        base_url=GROQ_ENDPOINT,
        api_key=GROQ_API_KEY,
    ), GROQ_MODEL


def _build_context(state: dict) -> str:
    stats   = state.get("stats", {})
    patterns = [p["name"] if isinstance(p, dict) else p for p in state.get("patterns", [])]
    tension  = state.get("tension", 0)
    return (
        f"Match minute: {stats.get('minute', '?')}\n"
        f"Tension score: {tension:.1f}/100\n"
        f"Recent passes: {stats.get('passes', 0)} "
        f"(accuracy {stats.get('pass_accuracy', 0):.0f}%)\n"
        f"Recent shots: {stats.get('shots', 0)}\n"
        f"Pressure events: {stats.get('pressures', 0)}\n"
        f"% events in danger zone: {stats.get('in_danger_zone_pct', 0):.1f}%\n"
        f"Forward pass rate: {stats.get('forward_pass_rate', 0):.0f}%\n"
        f"Active tactical patterns: {', '.join(patterns) if patterns else 'None detected'}"
    )


ANALYST_PROMPT = """\
You are an elite football tactical analyst commentating live.
Based on this real-time match state, write exactly 2 sentences:
1. What tactical pattern is building and why the tension score is what it is
2. What specifically to watch for in the next 3-5 minutes

Be precise — cite numbers, patterns, and spatial cues. No fluff.

Match state:
{context}"""

FAN_PROMPT = """\
You are an exciting football commentator speaking to a casual fan.
Based on this match state, write 1-2 short, punchy sentences that:
- Build excitement and anticipation
- Explain what's happening in plain English (no jargon)
- Make them feel like something big is about to happen

Match state:
{context}"""


def narrate_analyst(state: dict) -> str:
    client, model = _get_client()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": ANALYST_PROMPT.format(context=_build_context(state))
            }],
            max_tokens=180,
            temperature=0.7,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"[Narrative unavailable: {e}]"


def narrate_fan(state: dict) -> str:
    client, model = _get_client()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": FAN_PROMPT.format(context=_build_context(state))
            }],
            max_tokens=100,
            temperature=0.9,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"[Narrative unavailable: {e}]"


def narrate_both(state: dict) -> dict:
    return {
        "analyst": narrate_analyst(state),
        "fan":     narrate_fan(state),
        "tension": state.get("tension", 0),
        "minute":  state.get("stats", {}).get("minute", 0),
    }
