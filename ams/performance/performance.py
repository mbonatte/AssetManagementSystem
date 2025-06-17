"""
Created on Sep 21, 2022.

@author: MauricioBonatte
@e-mail: mbonatte@ymail.com
"""

import numpy as np
from multiprocessing import Pool
from scipy.linalg import expm
from random import choices
from typing import List, Dict

from .maintenance import ActionEffect
from .hazard_effects import HazardEffect
from .windows import EffectWindow

class Sample():
    def __init__(self):
        self.state = None
        self.timeOfDelay = 0
        self.improvement = 0
        self.timeOfReduction = 0
        self.rateOfReduction = 1


class Performance():
    def __init__(self, deterioration_model, maintenance_actions, hazard_data=None):
        # — base deterioration model ------------------------------------------------
        self.deterioration_model = deterioration_model
        self.Q = self.deterioration_model.intensity_matrix
        self.standard_transition_matrix = expm(self.Q)

        self.list_of_possible_ICs = np.linspace(
            start=self.deterioration_model.best_IC,
            stop=self.deterioration_model.worst_IC,
            num=self.deterioration_model._number_of_states,
            dtype=int,
        )

        # — maintenance & hazard effects -------------------------------------------
        self.action_effects = ActionEffect.set_action_effects(maintenance_actions)
        self.hazard_effects = HazardEffect.set_hazard_effects(hazard_data or [])

        # schedules & runtime state
        self.actions_schedule = {}
        self._active_windows: List[EffectWindow] = []

    def _open_window(self, win: EffectWindow) -> None:
        """Register a new EffectWindow."""
        self._active_windows.append(win)

    def _purge_expired(self, t: int) -> None:
        """Drop windows that ended before time *t*."""
        self._active_windows = [w for w in self._active_windows if w.is_active(t)]

    def _current_factor(self, t: int) -> float:
        """
        Return the factor of the *most-recent* still-alive window.

        We iterate the list in reverse insertion order; the first active match
        is therefore the latest one opened.
        """
        self._purge_expired(t)
        for w in reversed(self._active_windows):
            if w.is_active(t):
                return w.factor
        return 1.0
    
    def get_action(self, time) -> List:
        return self.actions_schedule.get(str(time), None)

    def _set_actions_schedule(self, actions_schedule: Dict):
        self.actions_schedule = actions_schedule

    def get_reduction_factor(self, interventions: dict, time: int) -> float:
        """
        Simple, stateless computation that matches the unit-test expectations
        :contentReference[oaicite:0]{index=0}.
        """
        factor = 1.0
        for t, interv in interventions.items():
            if t > time:
                continue
            elapsed = time - t

            # suppression window (delay) dominates
            if elapsed < interv.timeOfDelay:
                return 0.0

            # reduction window – keep the strictest (lowest) factor
            if elapsed < interv.timeOfReduction:
                factor = min(factor, interv.rateOfReduction)

        return factor
    
    def get_increase_factor(self, time, hazard, current_state):
        self.hazard_active = True
        self.intervertion_active = False

        self.last_hazard.update({
            "time": time,
            "duration": hazard.get_time_of_increase(current_state),
            "reduction": hazard.get_increase_rate(current_state)
        })

        return hazard.get_increase_rate(current_state)

    def _choose_randomly_the_next_IC(self, current_IC, transition_matrix):
        # Calculate the index for the current_IC relative to the best_IC
        IC_index = abs(current_IC - self.deterioration_model.best_IC)
        
        # Use the transition matrix to determine the probabilities
        prob = transition_matrix[IC_index]
        
        # Randomly select the next IC based on the calculated probabilities
        return choices(self.list_of_possible_ICs, prob, k=1)[0]

    def get_improved_IC(self, IC, improvement):
        if self.deterioration_model._is_transition_crescent:
            return max(IC - improvement, self.deterioration_model.best_IC)
        else:
            return min(IC + improvement, self.deterioration_model.best_IC)
        
    def get_degradated_IC(self, IC, degradation):
        if self.deterioration_model._is_transition_crescent:
            return min(IC + degradation, self.deterioration_model.worst_IC)
        else:
            return max(IC - degradation, self.deterioration_model.worst_IC)

    def compute_modified_transition_matrix(self, reduction_factor):
        if reduction_factor == 1:
            return self.standard_transition_matrix
        return expm(self.Q * reduction_factor)
    
    def _schedule_action(self, start: int, interv: Sample) -> None:
        """
        Convert *interv* into EffectWindow objects **and** discard any action
        windows that were opened by older maintenance actions.

        This matches the legacy behaviour where the dictionaries
        `self.last_intervention/*` always held *only* the most-recent action.
        """
        # ❶ remove windows generated by earlier actions
        self._active_windows = [
            w for w in self._active_windows if w.priority != 1
        ]

        # ❷ (re)insert suppression / reduction windows for the *current* action
        if interv.timeOfDelay:
            self._open_window(EffectWindow(start, interv.timeOfDelay, 0.0))

        if (
            interv.timeOfReduction
            and interv.rateOfReduction < 1.0
        ):
            self._open_window(
                EffectWindow(
                    start + interv.timeOfDelay,
                    interv.timeOfReduction,
                    interv.rateOfReduction,
                )
            )
    
    def _schedule_hazard(
        self, start: int, hazard: HazardEffect, state: int
    ) -> None:
        if not hazard.increase_rate:
            return
        dur = hazard.get_time_of_increase(state)
        fac = hazard.get_increase_rate(state)
        if dur and fac > 1.0:
            self._open_window(EffectWindow(start, dur, fac))

    def _get_next_IC(self,
                     current_state: int,
                     interventions: dict,
                     time: int,
                     hazards):
        """
        Get the next state by unit time.

        Considering the past and actual interventions.

        Parameters
        ----------
        current_state : TYPE
            DESCRIPTION.
        interventions : TYPE
            DESCRIPTION.
        time : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        #hazard
        hazard_name = hazards.get(time)
        hazard = self.hazard_effects.get(hazard_name)
        
        if hazard and hazard.degradation:
            return self.get_degradated_IC(current_state, hazard.get_degradation(current_state))
        
        interv = interventions.get(time)
        if interv and interv.improvement:
            return self.get_improved_IC(current_state, interv.improvement)
        
        
        if interv:
            self._schedule_action(time, interv)
        if hazard and hazard.increase_rate:
            self._schedule_hazard(time, hazard, current_state)
        
        factor = self._current_factor(time)
        if factor == 0.0:                 # suppression
            return current_state

        # compute transition matrix
        transition_matrix = self.compute_modified_transition_matrix(factor)

        # calculate next state
        next_IC = self._choose_randomly_the_next_IC(current_state,
                                                    transition_matrix)
        return next_IC

    def set_interventions_effect(
        self,
        intervention: Sample,
        action: ActionEffect,
        IC: int,
        start_time: int | None = None,
    ) -> None:
        """Populate *intervention* fields and open its windows (if start_time)."""
        s = int(IC)
        intervention.timeOfDelay = action.get_time_of_delay(s)
        intervention.improvement = action.get_improvement(s)
        intervention.timeOfReduction = action.get_time_of_reduction(s)
        intervention.rateOfReduction = action.get_reduction_rate(s)

        if start_time is not None:
            self._schedule_action(start_time, intervention)
  
    def _sample_hazards(self, horizon: int) -> Dict[int, str]:
        """Draws a synthetic hazards schedule of length *horizon*."""
        damages = list(self.hazard_effects.keys())
        probs = np.array([self.hazard_effects[d].probability for d in damages], dtype=float)
        probs /= probs.sum()
        samples = np.random.choice(damages, p=probs, size=horizon)
        return {i + 1: d for i, d in enumerate(samples) if d != "No Damage"}
    
    def predict_MC(
            self,
            time_horizon: int,
            initial_IC: int,
            hazards_schedule: Dict[int, str] = None,
        ) -> np.ndarray:
        self._active_windows = []

        #Time horizon for years [0 + time_horizon]
        time_horizon += 1
        asset_condition = np.empty(time_horizon, dtype=int)
        asset_condition[0] = initial_IC
        
        if hazards_schedule is None:
            hazards_schedule = self._sample_hazards(time_horizon)

        interventions = {
            int(year): Sample()
            for year in self.actions_schedule
        }
        
        ## Cache the method reference for better performance
        for time in range(1, time_horizon):
            action_name = self.get_action(time)
            if action_name and action_name in self.action_effects:
                self.set_interventions_effect(
                    interventions[time],
                    self.action_effects[action_name],
                    asset_condition[time-1],
                    start_time=time,
                )
            
            asset_condition[time] = self._get_next_IC(
                asset_condition[time-1], interventions, time, hazards_schedule
            )
        return asset_condition

    def get_IC_over_time(self,
                         time_horizon: int,
                         initial_IC: int = None,
                         actions_schedule: Dict = None,
                         hazards_schedule: Dict = None,
                         number_of_samples:int = 10) -> np.array:
        """
        Get the mean prediction by Monte Carlo approach.

        Parameters
        ----------
        time_horizon : int
            Time horizon.
        initial_IC : int, optional
            The initial condition index. The default is None.
        number_of_samples : TYPE, optional
            Number of samples in the Monte Carlo. The default is 100.

        Returns
        -------
        np.array
            Mean performance over time.

        """
        if initial_IC is None:
            initial_IC = self.deterioration_model.best_IC
        
        self._set_actions_schedule(actions_schedule or {})
        
        samples = [
            self.predict_MC(time_horizon,initial_IC,hazards_schedule)
            for _ in range(number_of_samples)
        ]
        return np.mean(samples, axis=0)  # Mean pear year