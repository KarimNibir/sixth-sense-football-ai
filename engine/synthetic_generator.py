"""
Synthetic Football Event Generator
Produces realistic match event streams based on football probability models.
Satisfies hackathon Category (v): Data Innovation.

Each event mirrors the StatsBomb schema so the rest of the pipeline
works identically on synthetic and real data.
"""
import random, time, math, uuid
from dataclasses import dataclass, field, asdict
from typing import Optional
import json

# ─── Pitch dimensions (StatsBomb standard) ───────────────────────────────────
PITCH_LENGTH = 120.0
PITCH_WIDTH  = 80.0

# ─── Team templates ──────────────────────────────────────────────────────────
TEAMS = {
    "Manchester City": {
        "style": "possession",        # high press, short passes
        "press_intensity": 0.75,
        "pass_forward_bias": 0.45,
        "shot_frequency": 0.08,
        "players": [
            "Ederson","Kyle Walker","Rúben Dias","Manuel Akanji","Josko Gvardiol",
            "Rodri","Kevin De Bruyne","Bernardo Silva",
            "Phil Foden","Erling Haaland","Jeremy Doku"
        ]
    },
    "Arsenal": {
        "style": "pressing",
        "press_intensity": 0.80,
        "pass_forward_bias": 0.50,
        "shot_frequency": 0.09,
        "players": [
            "David Raya","Ben White","William Saliba","Gabriel Magalhães","Oleksandr Zinchenko",
            "Thomas Partey","Martin Ødegaard","Declan Rice",
            "Bukayo Saka","Leandro Trossard","Gabriel Martinelli"
        ]
    }
}

EVENT_TYPES = ["Pass","Carry","Pressure","Shot","Tackle","Interception",
               "Ball Receipt","Dribble","Clearance","Foul Committed"]

PASS_OUTCOMES     = ["Complete","Incomplete","Out","Pass Offside"]
SHOT_OUTCOMES     = ["Goal","Saved","Off T","Blocked","Wayward"]
DRIBBLE_OUTCOMES  = ["Complete","Incomplete"]
TACKLE_OUTCOMES   = ["Won","Lost"]

@dataclass
class MatchEvent:
    id:             str
    index:          int
    minute:         int
    second:         int
    type:           str
    team:           str
    player:         str
    location:       list
    possession:     int
    possession_team: str

    # optional fields
    pass_end_location:      Optional[list] = None
    pass_outcome:           Optional[str]  = None
    pass_length:            Optional[float]= None
    pass_angle:             Optional[float]= None
    shot_outcome:           Optional[str]  = None
    shot_statsbomb_xg:      Optional[float]= None
    under_pressure:         Optional[bool] = None
    carry_end_location:     Optional[list] = None
    dribble_outcome:        Optional[str]  = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


class SyntheticMatchGenerator:
    """
    Generates a realistic synthetic football match as a stream of events.
    Probabilities are derived from StatsBomb open-data averages.
    """

    def __init__(self, home_team: str = "Manchester City",
                 away_team: str = "Arsenal",
                 seed: int = None):
        self.home_team   = home_team
        self.away_team   = away_team
        self.teams       = [home_team, away_team]
        self.team_data   = TEAMS
        if seed: random.seed(seed)

        # match state
        self.minute      = 0
        self.second      = 0
        self.index       = 0
        self.possession  = 1
        self.poss_team   = home_team
        self.score       = {home_team: 0, away_team: 0}
        self.events      = []

    # ── helpers ──────────────────────────────────────────────────────────────

    def _next_player(self, team: str) -> str:
        players = TEAMS.get(team, {}).get("players", ["Player"])
        return random.choice(players)

    def _advance_time(self, base_seconds: int = 3):
        """Move match clock forward by a realistic amount."""
        self.second += base_seconds + random.randint(0, 4)
        while self.second >= 60:
            self.second -= 60
            self.minute += 1

    def _zone(self, x: float) -> str:
        if x < 40:  return "Defensive Third"
        if x < 80:  return "Middle Third"
        return "Attacking Third"

    def _random_location(self, zone: str = "Middle Third") -> list:
        if zone == "Defensive Third":
            x = random.uniform(2, 40)
        elif zone == "Attacking Third":
            x = random.uniform(80, 118)
        else:
            x = random.uniform(40, 80)
        y = random.uniform(2, 78)
        return [round(x, 1), round(y, 1)]

    def _make_event(self, etype: str, team: str, location: list,
                    **kwargs) -> MatchEvent:
        self.index += 1
        ev = MatchEvent(
            id=str(uuid.uuid4()),
            index=self.index,
            minute=self.minute,
            second=self.second,
            type=etype,
            team=team,
            player=self._next_player(team),
            location=location,
            possession=self.possession,
            possession_team=self.poss_team,
            **kwargs
        )
        return ev

    # ── event generators ─────────────────────────────────────────────────────

    def _gen_pass(self, team: str, location: list) -> MatchEvent:
        tdata      = TEAMS.get(team, {})
        fwd_bias   = tdata.get("pass_forward_bias", 0.45)
        angle      = random.gauss(0 if random.random() < fwd_bias else math.pi, 0.8)
        length     = random.lognormvariate(2.8, 0.6)   # ~15m avg
        length     = min(length, 80)
        ex = location[0] + length * math.cos(angle)
        ey = location[1] + length * math.sin(angle)
        ex = max(1, min(119, ex))
        ey = max(1, min(79, ey))

        # outcome probability
        p_complete = 0.82 if length < 20 else 0.65 if length < 40 else 0.50
        outcome = "Complete" if random.random() < p_complete else random.choice(["Incomplete","Out"])

        return self._make_event("Pass", team, location,
            pass_end_location=[round(ex,1), round(ey,1)],
            pass_outcome=outcome,
            pass_length=round(length,1),
            pass_angle=round(angle,3),
            under_pressure=random.random() < 0.25)

    def _gen_carry(self, team: str, location: list) -> MatchEvent:
        dx = random.gauss(5, 3)
        dy = random.gauss(0, 3)
        end = [round(min(119, max(1, location[0]+dx)),1),
               round(min(79,  max(1, location[1]+dy)),1)]
        return self._make_event("Carry", team, location, carry_end_location=end)

    def _gen_pressure(self, pressing_team: str, location: list) -> MatchEvent:
        return self._make_event("Pressure", pressing_team, location)

    def _gen_shot(self, team: str) -> MatchEvent:
        # shots happen near goal
        x = random.uniform(90, 118)
        y = random.uniform(25, 55)
        dist_to_goal = math.sqrt((120-x)**2 + (40-y)**2)
        xg = max(0.02, min(0.95, 0.5 * math.exp(-dist_to_goal / 12)))

        if random.random() < xg * 0.8:
            outcome = "Goal"
            self.score[team] += 1
        elif random.random() < 0.4:
            outcome = "Saved"
        elif random.random() < 0.5:
            outcome = "Off T"
        else:
            outcome = "Blocked"

        return self._make_event("Shot", team, [round(x,1), round(y,1)],
            shot_outcome=outcome,
            shot_statsbomb_xg=round(xg, 3))

    # ── possession sequence ───────────────────────────────────────────────────

    def _gen_possession_sequence(self, team: str) -> list:
        """Generate 3-15 events for one possession sequence."""
        events      = []
        opp         = self.away_team if team == self.home_team else self.home_team
        tdata       = TEAMS.get(team, {})
        shot_freq   = tdata.get("shot_frequency", 0.07)
        press_int   = TEAMS.get(opp, {}).get("press_intensity", 0.6)
        location    = self._random_location("Middle Third")
        seq_len     = random.randint(3, 15)

        for i in range(seq_len):
            self._advance_time()

            # opposition pressure?
            if random.random() < press_int * 0.3:
                press_loc = [location[0] + random.uniform(-3,3),
                             location[1] + random.uniform(-3,3)]
                events.append(self._gen_pressure(opp, press_loc))
                self._advance_time(1)

            # decide next action
            roll = random.random()

            if location[0] > 85 and roll < shot_freq * 2:
                # in attacking third — shoot
                ev = self._gen_shot(team)
                events.append(ev)
                if ev.shot_outcome in ("Goal", "Off T", "Blocked"):
                    # possession ends
                    self.possession += 1
                    self.poss_team = opp
                    break

            elif roll < 0.55:
                ev = self._gen_pass(team, location)
                events.append(ev)
                if ev.pass_outcome == "Complete":
                    location = ev.pass_end_location
                else:
                    # turnover
                    self.possession += 1
                    self.poss_team = opp
                    break

            elif roll < 0.80:
                ev = self._gen_carry(team, location)
                events.append(ev)
                location = ev.carry_end_location

            else:
                # tackle / lost duel
                events.append(self._make_event("Tackle", opp, location))
                self.possession += 1
                self.poss_team = opp
                break

        return events

    # ── full match ────────────────────────────────────────────────────────────

    def generate_match(self, duration_minutes: int = 90) -> list:
        """Generate a full synthetic match."""
        self.minute = 0
        all_events  = []

        while self.minute < duration_minutes:
            team   = self.poss_team
            seq    = self._gen_possession_sequence(team)
            all_events.extend(seq)

            # halftime
            if 44 <= self.minute <= 46 and not any(
                    e.minute == 45 and e.type == "Pass" for e in all_events[-5:]):
                self.minute = 45

        self.events = all_events
        return all_events

    def to_json(self) -> str:
        return json.dumps([e.to_dict() for e in self.events], indent=2)

    def summary(self) -> dict:
        goals  = [e for e in self.events if e.shot_outcome == "Goal"]
        passes = [e for e in self.events if e.type == "Pass"]
        shots  = [e for e in self.events if e.type == "Shot"]
        return {
            "total_events": len(self.events),
            "score": self.score,
            "passes": len(passes),
            "pass_accuracy": round(
                sum(1 for p in passes if p.pass_outcome=="Complete") / max(len(passes),1) * 100, 1),
            "shots": len(shots),
            "goals": len(goals),
            "possessions": self.possession
        }


if __name__ == "__main__":
    gen = SyntheticMatchGenerator("Manchester City", "Arsenal", seed=42)
    events = gen.generate_match(90)
    print(json.dumps(gen.summary(), indent=2))
    print(f"\nSample event:\n{json.dumps(events[10].to_dict(), indent=2)}")
