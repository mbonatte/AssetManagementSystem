"""
Created on Sep 21, 2022.

@author: MauricioBonatte
@e-mail: mbonatte@ymail.com

maintenance.py - Classes for maintenance actions and effects.
"""

from typing import List, Dict
import numpy as np

class ActionEffect():
    """Class for effects of maintenance actions."""

    def set_action_effects(actions: Dict[str, Dict]
    ) -> Dict[str, 'ActionEffect']:
        """Create ActionEffect from dictionary."""
        
        effects = {'DoNothing': ActionEffect('DoNothing')}
        
        for action in actions:
            effect = ActionEffect(action['name'])
            
            for key in ['delay', 'improvement', 'time_of_reduction', 'reduction_rate']:
                if key in action:
                    if key == 'delay':
                        effect.set_time_of_delay(action[key])
                    if key == 'improvement':
                        effect.set_improvement(action[key])
                    if key == 'time_of_reduction':
                        effect.set_time_of_reduction(action[key])
                    if key == 'reduction_rate':
                        effect.set_reduction_rate(action[key])
            
            effect.cost = action.get('cost', 0)  
            effects[effect.name] = effect
        return effects

    def __init__(self, name: str) -> None:
        """Initialize ActionEffect."""
                 
        self.name = name
        self.cost = 0
        
        # Effects modeled as triangular distribution
        self.time_of_delay = {}
        self.improvement = {}
        self.time_of_reduction = {}
        self.reduction_rate = {}

    def _get_effect(self, action):
        left = action[0]
        mode = action[1]
        right = action[2]
        try:
            return np.random.triangular(left, mode, right)
        except ValueError:
            return mode

    def get_improvement(self, state):
        return self._get_effect(self.improvement.get(state,[0,0,0]))

    def get_reduction_rate(self, state):
        return self._get_effect(self.reduction_rate.get(state,[1,1,1]))

    def get_time_of_delay(self, state):
        return self._get_effect(self.time_of_delay.get(state,[0,0,0]))

    def get_time_of_reduction(self, state):
        return self._get_effect(self.time_of_reduction.get(state,[0,0,0]))

    def _set_effect(self, action, effect):
        for i, eff in effect.items():
            try:
                action[int(i)] = eff
            except ValueError:
                continue

    def set_improvement(self, effect):
        return self._set_effect(self.improvement, effect)

    def set_reduction_rate(self, effect):
        return self._set_effect(self.reduction_rate, effect)

    def set_time_of_delay(self, effect):
        return self._set_effect(self.time_of_delay, effect)

    def set_time_of_reduction(self, effect):
        return self._set_effect(self.time_of_reduction, effect)
