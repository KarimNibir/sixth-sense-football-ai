"""
FastAPI server — real-time match intelligence API.
Serves the live dashboard and SSE event stream.
"""
import os, json, time, asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine.synthetic_generator import SyntheticMatchGenerator
from engine.pattern_engine import TensionCalculator
from engine.narrator import narrate_both

app = FastAPI(title="The Sixth Sense — Football AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Global match state ────────────────────────────────────────────────────────
match_state = {
    "running": False,
    "minute": 0,
    "events_processed": 0,
    "tension": 0,
    "patterns": [],
    "score": {},
    "last_insight": {"analyst": "", "fan": ""},
    "tension_history": [],
    "recent_events": []
}
calculator = TensionCalculator()
all_events = []
event_index = 0

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "../dashboard/index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return f.read()
    return "<h1>Dashboard not found</h1>"

@app.post("/match/start")
async def start_match(home: str = "Manchester City", away: str = "Arsenal"):
    global all_events, event_index, match_state, calculator
    gen = SyntheticMatchGenerator(home, away, seed=int(time.time()))
    all_events  = gen.generate_match(90)
    event_index = 0
    calculator  = TensionCalculator()
    match_state.update({
        "running": True, "minute": 0, "events_processed": 0,
        "tension": 0, "patterns": [], "score": gen.score,
        "last_insight": {"analyst": "Match starting...", "fan": "Kick off! 🏟️"},
        "tension_history": [], "recent_events": [],
        "home": home, "away": away, "total_events": len(all_events)
    })
    return {"status": "started", "total_events": len(all_events),
            "home": home, "away": away}

@app.get("/match/state")
async def get_state():
    return match_state

@app.post("/match/tick")
async def tick(events_per_tick: int = 5):
    """Advance match by N events — call repeatedly to simulate live feed."""
    global event_index, match_state
    if not match_state["running"] or event_index >= len(all_events):
        match_state["running"] = False
        return {"status": "finished"}

    # process next batch
    batch = all_events[event_index: event_index + events_per_tick]
    event_index += events_per_tick

    for ev in batch:
        calculator.push(ev.to_dict() if hasattr(ev, "to_dict") else ev)

    result = calculator.compute()
    tension = result["tension"]

    # generate narrative every 15 events or when tension spikes
    should_narrate = (
        event_index % 15 == 0 or
        abs(tension - match_state["tension"]) > 15 or
        any(p["severity"] == "critical" for p in result["patterns"])
    )

    if should_narrate and os.environ.get("GROQ_API_KEY"):
        try:
            insight = narrate_both(result)
        except:
            insight = match_state["last_insight"]
    else:
        insight = match_state["last_insight"]

    # update score from events
    goals = [e for e in batch
             if (hasattr(e,'shot_outcome') and e.shot_outcome=="Goal") or
                (isinstance(e,dict) and e.get("shot_outcome")=="Goal")]
    score = dict(match_state.get("score", {}))
    for g in goals:
        team = g.team if hasattr(g,'team') else g.get("team","")
        if team: score[team] = score.get(team,0) + 1

    # tension history
    history = match_state["tension_history"][-89:]
    history.append({"minute": result["stats"]["minute"], "tension": tension})

    # recent events for display
    recent = [
        {"type": (e.type if hasattr(e,'type') else e.get("type","")),
         "team": (e.team if hasattr(e,'team') else e.get("team","")),
         "player": (e.player if hasattr(e,'player') else e.get("player","")),
         "minute": (e.minute if hasattr(e,'minute') else e.get("minute",0))}
        for e in batch[-5:]
    ]

    match_state.update({
        "tension": tension,
        "patterns": result["patterns"],
        "events_processed": event_index,
        "minute": result["stats"]["minute"],
        "score": score,
        "last_insight": insight,
        "tension_history": history,
        "recent_events": recent,
        "stats": result["stats"]
    })

    return match_state

@app.get("/match/stream")
async def stream():
    """Server-Sent Events stream — auto-advances match and pushes updates."""
    async def event_generator():
        while match_state.get("running", False) and event_index < len(all_events):
            await asyncio.sleep(1.5)   # 1.5s per tick = ~2 min of match per minute
            # auto-tick
            batch = all_events[event_index: event_index+8] if event_index < len(all_events) else []
            if not batch:
                break
            # (tick logic duplicated for SSE — in production extract to shared fn)
            for ev in batch:
                calculator.push(ev.to_dict() if hasattr(ev,'to_dict') else ev)
            result    = calculator.compute()
            data      = json.dumps({"tension": result["tension"],
                                    "patterns": result["patterns"],
                                    "stats": result["stats"],
                                    "minute": result["stats"]["minute"]})
            yield f"data: {data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
