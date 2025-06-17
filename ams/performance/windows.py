# windows.py
from dataclasses import dataclass


@dataclass(slots=True)
class EffectWindow:
    start: int          # timestep when the window becomes active
    duration: int       # how many steps it lasts
    factor: float       # 0 => suppression, <1 => reduction, >1 => acceleration
    priority: int = 1   # kept for future use—but no longer used to decide

    def is_active(self, t: int) -> bool:
        return self.start <= t < self.start + self.duration
