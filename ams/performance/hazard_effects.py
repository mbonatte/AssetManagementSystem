"""
Created on Nov 22, 2024.

@author: MauricioBonatte
@e-mail: mbonatte@ymail.com

hazard.py - Classes for hazard effects.
"""

from typing import Dict
import numpy as np

class HazardEffect:
    """Class for effects of hazards."""

    def set_hazard_effects(hazards: list) -> dict:
        """
        Create HazardEffect from a list of dictionaries and consider probabilities.
        Add 'No Damage' scenario with remaining probability.
        """
        effects = {}
        total_probability = sum(hazard['probability'] for hazard in hazards)

        # Ensure 'No Damage' has the remaining probability
        remaining_probability = max(0, 1 - total_probability)

        # Add 'No Damage' scenario with the remaining probability
        no_damage_effect = HazardEffect('No Damage')
        no_damage_effect.set_probability(remaining_probability)
        effects = {'No Damage': no_damage_effect}
        
        # Add each hazard effect
        for hazard in hazards:
            effect = HazardEffect(hazard['name'])
            
            # Set properties based on keys in the hazard dictionary
            for key in ['delay', 'degradation', 'time_of_increase', 'increase_rate', 'probability']:
                if key in hazard:
                    if key == 'delay':
                        effect.set_time_of_delay(hazard[key])
                    if key == 'degradation':
                        effect.set_degradation(hazard[key])
                    if key == 'time_of_increase':
                        effect.set_time_of_increase(hazard[key])
                    if key == 'increase_rate':
                        effect.set_increase_rate(hazard[key])
                    if key == 'probability':
                        effect.set_probability(hazard[key])
            
            # Add the effect to the effects dictionary
            effects[effect.name] = effect

        return effects

    def __init__(self, name: str) -> None:
        """Initialize HazardEffect."""
        
        self.name = name
        self.probability = 0
        
        # Effects modeled as triangular distribution
        self.time_of_delay = {}
        self.degradation = {}
        self.time_of_increase = {}
        self.increase_rate = {}

    def _get_effect(self, action):
        left = action[0]
        mode = action[1]
        right = action[2]
        try:
            return np.random.triangular(left, mode, right)
        except ValueError:
            return mode

    def get_degradation(self, state):
        return self._get_effect(self.degradation.get(state, [0, 0, 0]))

    def get_increase_rate(self, state):
        return self._get_effect(self.increase_rate.get(state, [1, 1, 1]))

    def get_time_of_delay(self, state):
        return self._get_effect(self.time_of_delay.get(state, [0, 0, 0]))

    def get_time_of_increase(self, state):
        return self._get_effect(self.time_of_increase.get(state, [0, 0, 0]))

    def _set_effect(self, action, effect):
        for i, eff in effect.items():
            try:
                action[int(i)] = eff
            except ValueError:
                continue

    def set_probability(self, prob):
        self.probability = prob
    
    def set_degradation(self, effect):
        self._set_effect(self.degradation, effect)

    def set_increase_rate(self, effect):
        self._set_effect(self.increase_rate, effect)

    def set_time_of_delay(self, effect):
        self._set_effect(self.time_of_delay, effect)

    def set_time_of_increase(self, effect):
        self._set_effect(self.time_of_increase, effect)
