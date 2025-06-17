"""
Created on Sep 21, 2022.

@author: MauricioBonatte
@e-mail: mbonatte@ymail.com

maintenance.py - Classes for maintenance actions and effects.
"""

from typing import List, Dict
import numpy as np

from .effect_base import EffectBase

class ActionEffect(EffectBase):
    """Class for effects of maintenance actions."""

    def set_action_effects(actions: Dict[str, Dict]
    ) -> Dict[str, 'ActionEffect']:
        """Create ActionEffect from dictionary."""
        
        effects = {'DoNothing': ActionEffect('DoNothing')}

        # Mapping from action keys to internal effect keys
        key_map = {
            'delay': 'time_of_delay',
            'improvement': 'improvement',
            'time_of_reduction': 'time_of_reduction',
            'reduction_rate': 'reduction_rate',
        }
        
        for action in actions:
            effect = ActionEffect(action['name'])
            for src_key, dest_key in key_map.items():
                if src_key in action:
                    effect.set(dest_key, action[src_key])
            
            effect.cost = action.get('cost', 0)  
            effects[effect.name] = effect
        return effects

    _FIELDS = [
        "improvement",
        "time_of_delay",
        "reduction_rate",
        "time_of_reduction"
    ]
    
    def __init__(self, name: str) -> None:
        """Initialize ActionEffect."""
        super().__init__(name)
        self.cost: float = 0.0

    def get_improvement(self, state):       return self.get("improvement", state, [0,0,0])
    def get_reduction_rate(self, state):    return self.get("reduction_rate", state, [1,1,1])
    def get_time_of_delay(self, state):     return self.get("time_of_delay", state, [0,0,0])
    def get_time_of_reduction(self, state): return self.get("time_of_reduction", state, [0,0,0])

    def set_improvement(self, eff): self.set("improvement", eff)
    def set_reduction_rate(self, eff): self.set("reduction_rate", eff)
    def set_time_of_delay(self, eff): self.set("time_of_delay", eff)
    def set_time_of_reduction(self, eff): self.set("time_of_reduction", eff)
