"""
Created on Nov 22, 2024.

@author: MauricioBonatte
@e-mail: mbonatte@ymail.com

hazard.py - Classes for hazard effects.
"""

from typing import Dict
import numpy as np

from .effect_base import EffectBase

class HazardEffect(EffectBase):
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
            for key in ['degradation', 'time_of_increase', 'increase_rate', 'probability']:
                if key in hazard:
                    if key != 'probability':
                        effect.set(key, hazard[key])
                    if key == 'probability':
                        effect.set_probability(hazard[key])
            
            # Add the effect to the effects dictionary
            effects[effect.name] = effect

        return effects

    _FIELDS = [
        "degradation",
        "increase_rate",
        "time_of_increase"
    ]
    
    def __init__(self, name: str) -> None:
        """Initialize HazardEffect."""
        super().__init__(name)
        self.probability: float = 0.0

    def get_degradation(self, state):       return self.get("degradation", state, [0,0,0])
    def get_increase_rate(self, state):     return self.get("increase_rate", state, [1, 1, 1])
    def get_time_of_increase(self, state):  return self.get("time_of_increase", state, [0,0,0])

    def set_degradation(self, eff): self.set("degradation", eff)
    def set_increase_rate(self, eff): self.set("increase_rate", eff)
    def set_time_of_increase(self, eff): self.set("time_of_increase", eff)

    def set_probability(self, prob):
        self.probability = prob
