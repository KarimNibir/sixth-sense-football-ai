"""
Pattern Engine — real-time sequence encoder + tension calculator.
Converts a rolling window of match events into a tension score (0-100)
and a set of detected tactical patterns.
"""
import math, numpy as np
from collections import deque
from typing import List, Dict, Any

WINDOW_SIZE = 15  # events to analyse at once

# ─── Feature extraction ───────────────────────────────────────────────────────

def encode_sequence(events: List[Dict]) -> np.ndarray:
    """
    Convert a list of raw event dicts into a 40-dim feature vector.
    Works on both synthetic and real StatsBomb events.
    """
    if not events:
        return np.zeros(40)

    passes      = [e for e in events if e.get("type") == "Pass"]
    shots       = [e for e in events if e.get("type") == "Shot"]
    carries     = [e for e in events if e.get("type") == "Carry"]
    pressures   = [e for e in events if e.get("type") == "Pressure"]
    tackles     = [e for e in events if e.get("type") in ("Tackle","Interception")]

    n = max(len(events), 1)

    # ── possession features ──────────────────────────────────────────────────
    teams       = [e.get("team","") for e in events]
    team_counts = {}
    for t in teams:
        team_counts[t] = team_counts.get(t, 0) + 1
    top_team    = max(team_counts, key=team_counts.get, default="")
    possession_dominance = team_counts.get(top_team, 0) / n   # 0-1

    # ── spatial features ─────────────────────────────────────────────────────
    locs = [e["location"] for e in events if "location" in e and e["location"]]
    xs   = [l[0] for l in locs if isinstance(l, list) and len(l)==2]
    ys   = [l[1] for l in locs if isinstance(l, list) and len(l)==2]

    avg_x     = np.mean(xs) / 120 if xs else 0.5
    avg_y     = np.mean(ys) / 80  if ys else 0.5
    x_spread  = np.std(xs) / 60   if len(xs) > 1 else 0
    y_spread  = np.std(ys) / 40   if len(ys) > 1 else 0

    # spatial progression: is play moving toward goal?
    x_prog = 0.0
    if len(xs) > 2:
        x_prog = (xs[-1] - xs[0]) / 120

    # proximity to danger zone (x > 85)
    danger_events = sum(1 for x in xs if x > 85) / max(len(xs), 1)

    # ── pass features ────────────────────────────────────────────────────────
    fwd_passes   = sum(1 for p in passes
                       if isinstance(p.get("pass_end_location"), list)
                       and isinstance(p.get("location"), list)
                       and len(p["pass_end_location"])==2
                       and p["pass_end_location"][0] > p["location"][0])
    back_passes  = sum(1 for p in passes
                       if isinstance(p.get("pass_end_location"), list)
                       and isinstance(p.get("location"), list)
                       and len(p["pass_end_location"])==2
                       and p["pass_end_location"][0] < p["location"][0] - 5)

    pass_total      = max(len(passes), 1)
    fwd_rate        = fwd_passes / pass_total
    back_rate       = back_passes / pass_total
    pass_rate       = len(passes) / n

    complete_passes = sum(1 for p in passes if p.get("pass_outcome") == "Complete")
    pass_accuracy   = complete_passes / pass_total

    avg_pass_len    = 0.0
    lens = [p.get("pass_length", 0) for p in passes if p.get("pass_length")]
    if lens: avg_pass_len = np.mean(lens) / 80

    under_pressure_passes = sum(1 for p in passes if p.get("under_pressure")) / pass_total

    # ── press / intensity features ───────────────────────────────────────────
    press_rate      = len(pressures) / n
    tackle_rate     = len(tackles) / n
    shot_rate       = len(shots) / n

    # ── possession change rate ───────────────────────────────────────────────
    poss_teams = [e.get("possession_team","") for e in events]
    changes    = sum(1 for i in range(1, len(poss_teams))
                     if poss_teams[i] != poss_teams[i-1])
    change_rate = changes / max(len(poss_teams)-1, 1)

    # ── time / tempo features ────────────────────────────────────────────────
    minutes = [e.get("minute", 0) for e in events]
    seconds = [e.get("second", 0) for e in events]
    times   = [m*60+s for m,s in zip(minutes, seconds)]
    tempo   = 0.0
    if len(times) > 1:
        elapsed = max(times[-1] - times[0], 1)
        tempo   = min(n / elapsed, 1.0)   # events per second, capped

    match_time_norm = (np.mean(minutes) / 90) if minutes else 0.5

    # ── xG accumulation ─────────────────────────────────────────────────────
    total_xg = sum(s.get("shot_statsbomb_xg", 0) or 0 for s in shots)

    # ── assemble 40-dim vector ───────────────────────────────────────────────
    vec = np.array([
        # possession (0-4)
        possession_dominance, change_rate, tempo, match_time_norm, pass_rate,
        # spatial (5-10)
        avg_x, avg_y, x_spread, y_spread, x_prog, danger_events,
        # passing (11-18)
        fwd_rate, back_rate, pass_accuracy, avg_pass_len,
        under_pressure_passes, complete_passes/pass_total,
        len(passes)/n, min(total_xg, 1.0),
        # pressing (19-23)
        press_rate, tackle_rate, shot_rate,
        len(pressures)/n, len(carries)/n,
        # type distributions (24-30)
        len(passes)/n, len(shots)/n, len(carries)/n, len(pressures)/n,
        len(tackles)/n, changes/max(n,1), possession_dominance,
        # derived (31-39)
        fwd_rate * danger_events,           # attacking momentum
        press_rate * change_rate,           # chaos index
        back_rate * under_pressure_passes,  # defensive stress
        x_prog * avg_x,                     # forward progression
        pass_accuracy * fwd_rate,           # quality forward play
        shot_rate * danger_events,          # shot danger
        tempo * change_rate,                # match intensity
        total_xg / max(len(shots), 1),      # shot quality
        match_time_norm * danger_events,    # late danger
    ], dtype=np.float32)

    return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)


# ─── Tension calculator ───────────────────────────────────────────────────────

class TensionCalculator:
    """
    Maintains a rolling window of events and computes a tension score (0-100).
    Also detects named tactical patterns.
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window     = deque(maxlen=window_size)
        self.history    = []          # list of (vector, tension) for trend
        self.score      = 0.0

    def push(self, event: Dict) -> None:
        self.window.append(event)

    def compute(self) -> Dict[str, Any]:
        events = list(self.window)
        if len(events) < 3:
            return {"tension": 0, "patterns": [], "vector": None}

        vec      = encode_sequence(events)
        tension  = self._score_tension(vec, events)
        patterns = self._detect_patterns(vec, events)

        self.score = tension
        self.history.append({"tension": tension, "vector": vec.tolist()})
        if len(self.history) > 200:
            self.history.pop(0)

        return {
            "tension": round(tension, 1),
            "patterns": patterns,
            "vector": vec.tolist(),
            "window_size": len(events),
            "stats": self._window_stats(events, vec)
        }

    def _score_tension(self, vec: np.ndarray, events: List[Dict]) -> float:
        """
        Weighted combination of features that correlate with goal danger.
        Weights derived empirically from StatsBomb open data analysis.
        """
        spatial_danger   = vec[10]   # events in danger zone
        x_prog           = max(vec[9], 0)  # forward progression
        press_rate       = vec[19]
        change_rate      = vec[1]
        shot_rate        = vec[21]
        atk_momentum     = vec[31]
        chaos            = vec[32]
        shot_danger      = vec[35]
        tempo            = vec[6+2]  # index 8 = x_spread repurposed; vec[2]=tempo
        late_danger      = vec[39]

        raw = (
            spatial_danger   * 0.25 +
            x_prog           * 0.15 +
            shot_rate        * 0.20 +
            press_rate       * 0.10 +
            change_rate      * 0.08 +
            atk_momentum     * 0.10 +
            shot_danger      * 0.07 +
            late_danger      * 0.05
        )

        # boost for shots in window
        shots = [e for e in events if e.get("type") == "Shot"]
        if shots:
            raw += 0.3
        if any(s.get("shot_outcome") == "Goal" for s in shots):
            raw = min(raw + 0.4, 1.0)

        return min(100, raw * 100)

    def _detect_patterns(self, vec: np.ndarray, events: List[Dict]) -> List[Dict]:
        patterns = []

        # Pattern 1: High press sequence
        if vec[19] > 0.3 and vec[1] > 0.25:
            patterns.append({
                "id": "high_press",
                "name": "High Press Building",
                "description": "Intense pressing with frequent possession changes",
                "severity": "medium"
            })

        # Pattern 2: Forward momentum surge
        if vec[9] > 0.15 and vec[10] > 0.3:
            patterns.append({
                "id": "forward_surge",
                "name": "Forward Momentum Surge",
                "description": "Play rapidly advancing into the attacking third",
                "severity": "high"
            })

        # Pattern 3: Backpass under pressure (reset before attack)
        if vec[12] > 0.25 and vec[14] > 0.3:
            patterns.append({
                "id": "defensive_reset",
                "name": "Defensive Reset Under Pressure",
                "description": "Team recycling possession backward — may precede switch of play",
                "severity": "low"
            })

        # Pattern 4: Shot storm
        if vec[21] > 0.15:
            patterns.append({
                "id": "shot_storm",
                "name": "Shot Storm",
                "description": "Multiple shots in quick succession — goalkeeper under siege",
                "severity": "critical"
            })

        # Pattern 5: Counterattack signal
        if vec[1] > 0.4 and vec[9] > 0.2 and vec[2] > 0.15:
            patterns.append({
                "id": "counterattack",
                "name": "Counterattack Initiated",
                "description": "Fast transition after turnover — defence exposed",
                "severity": "critical"
            })

        # Pattern 6: Possession dominance in final third
        if vec[5] > 0.72 and vec[0] > 0.7:
            patterns.append({
                "id": "siege",
                "name": "Attacking Siege",
                "description": "Dominant possession deep in opponent territory",
                "severity": "high"
            })

        return patterns

    def _window_stats(self, events: List[Dict], vec: np.ndarray) -> Dict:
        passes  = [e for e in events if e.get("type") == "Pass"]
        shots   = [e for e in events if e.get("type") == "Shot"]
        press   = [e for e in events if e.get("type") == "Pressure"]
        teams   = list({e.get("team","") for e in events})
        mins    = [e.get("minute", 0) for e in events if e.get("minute") is not None]

        return {
            "minute": max(mins) if mins else 0,
            "passes": len(passes),
            "shots": len(shots),
            "pressures": len(press),
            "pass_accuracy": round(vec[13]*100, 1),
            "in_danger_zone_pct": round(vec[10]*100, 1),
            "teams_in_window": teams,
            "forward_pass_rate": round(vec[11]*100, 1),
        }
