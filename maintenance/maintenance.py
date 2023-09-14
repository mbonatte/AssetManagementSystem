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

    def set_action_effects(number_of_states: int, actions: Dict[str, Dict]
    ) -> Dict[str, 'ActionEffect']:
        """Create ActionEffect from dictionary."""
        
        effects = {'DoNothing': ActionEffect('DoNothing', number_of_states)}
        
        for action in actions:
            effect = ActionEffect(action['name'], number_of_states)
            
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

    def __init__(self, name: str, number_of_states: int) -> None:
        """Initialize ActionEffect."""
                 
        self.name = name
        self.number_of_states = number_of_states
        self.cost = 0
        
        # Effects modeled as triangular distribution
        self.time_of_delay = [[0, 0, 0] for i in range(number_of_states)]
        self.improvement = [[0, 0, 0] for i in range(number_of_states)]
        self.time_of_reduction = [[0, 0, 0] for i in range(number_of_states)]
        self.reduction_rate = [[1, 1, 1] for i in range(number_of_states)]

    def _get_effect(self, action):
        left = action[0]
        mode = action[1]
        right = action[2]
        try:
            return np.random.triangular(left, mode, right)
        except ValueError:
            return mode

    def get_improvement(self, state):
        return self._get_effect(self.improvement[state])

    def get_reduction_rate(self, state):
        return self._get_effect(self.reduction_rate[state])

    def get_time_of_delay(self, state):
        return self._get_effect(self.time_of_delay[state])

    def get_time_of_reduction(self, state):
        return self._get_effect(self.time_of_reduction[state])

    def _set_effect(self, action, effect):
        effect = np.array(effect).reshape(self.number_of_states,
                                          3)  # 3 for the triangular distribution
        for i, eff in enumerate(effect):
            try:
                action[i] = list(np.float_(eff))
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
