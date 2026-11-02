"""Parsing of /sim/task (JSON) into a TaskSpec. Formats: hackathon/docs/INTERFACES.md, section "Задача"."""
import json
from dataclasses import dataclass, field

SECONDS_PER_OBJECT = {"sort": 45.0, "tubes": 60.0}  # SCORING.md, section "Время"


@dataclass
class TaskSpec:
    task: str  # sort2 | sort3 | tubes
    kind: str  # sort | tubes
    variant: str = ""  # tubes: easy | standard | color | lying
    rule: str = ""  # tubes: any_slot | by_type | by_cap_color
    categories: dict = field(default_factory=dict)  # sort: category -> tray id
    trays: dict = field(default_factory=dict)  # sort: tray id -> colour
    objects: dict = field(default_factory=dict)  # sort: object name -> category
    tubes: dict = field(default_factory=dict)  # tubes: tube id -> {"type", "cap"}
    racks: dict = field(default_factory=dict)  # tubes: rack id -> {"where", "tube", "color", "slots"}
    raw: dict = field(default_factory=dict)

    @property
    def n_items(self):
        return len(self.tubes) if self.kind == "tubes" else len(self.objects)

    @property
    def time_limit(self):
        """Sim seconds the evaluator gives for this layout."""
        return SECONDS_PER_OBJECT[self.kind] * self.n_items

    def hood_racks_for(self, tube_type, cap):
        """Racks in the fume hood that accept a tube of this type / cap colour under the task rule."""
        hood = {rid: r for rid, r in self.racks.items() if r.get("where") == "hood"}
        if self.rule == "by_type":
            return [rid for rid, r in hood.items() if r.get("tube") == tube_type]
        if self.rule == "by_cap_color":
            return [rid for rid, r in hood.items() if r.get("color") == cap]
        return list(hood)  # any_slot


def parse_task(text):
    d = json.loads(text)
    task = d.get("task", "")
    kind = "tubes" if task == "tubes" else "sort"
    return TaskSpec(
        task=task, kind=kind, variant=d.get("variant", ""), rule=d.get("rule", ""),
        categories=d.get("categories", {}), trays=d.get("trays", {}), objects=d.get("objects", {}),
        tubes=d.get("tubes", {}), racks=d.get("racks", {}), raw=d,
    )
