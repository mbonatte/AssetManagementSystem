"""
Created on Sep 21, 2022.

@author: MauricioBonatte
@e-mail: mbonatte@ymail.com
"""

import json
from typing import List, Dict

import numpy as np

class ActionEffect():

    def set_action_effects(number_of_states: int,
                           actions: Dict):
        action_effects = {}
        # Append a 'Do Nothing' action
        action_effects['DoNothing'] = ActionEffect('DoNothing',
                                                    number_of_states)
        for action in actions:
            action_effect = ActionEffect(action['name'],
                                            number_of_states)
            try:
                action_effect.set_time_of_delay(action['time_of_delay'])
            except KeyError:
                pass
            try:
                action_effect.set_improvement(action['improvement'])
            except KeyError:
                pass
            try:
                action_effect.set_time_of_reduction(
                    action['time_of_reduction'])
            except KeyError:
                pass
            try:
                action_effect.set_reduction_rate(action['reduction_rate'])
            except KeyError:
                pass
            action_effect.cost = action.get('cost', 0)
            action_effects[action_effect.name] = action_effect
        return action_effects

    def __init__(self,
                 name,
                 number_of_states
                 ):
        self.name = name
        self.number_of_states = number_of_states
        self.cost = 0
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
