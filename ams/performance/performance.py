"""
Created on Sep 21, 2022.

@author: MauricioBonatte
@e-mail: mbonatte@ymail.com
"""

from .maintenance import ActionEffect

import numpy as np
from multiprocessing import Pool
from scipy.linalg import expm
from random import choices
from typing import List, Dict

# from numba import jit

# @jit(nopython=True, nogil=True, cache=True)
# def numba_matrix_power(a, b):
    # return np.linalg.matrix_power(a, b)


# # Create my own 'expm', it's faster than scipy.linalg.expm
# import math
# factorials = [1/math.factorial(i) for i in range(15)]
# def expm(Q):
    # Q = [numba_matrix_power(Q,i) for i in range(6)]
    # P = sum(f*q for q,f in zip(Q,factorials))
    # return P

# # Create my own 'choices', it's faster than random.choices
# @jit(nopython=True, cache=True)
# def choices(population, weights):
    # r = np.random.rand()
    # for i in range(len(weights)-1):
        # A = 0
        # for j in range(i+1):
            # A += weights[j]
            # if(r < A):
                # return population[i]
    # return population[-1]

class Sample():
    def __init__(self):
        self.state = None
        self.timeOfDelay = 0
        self.improvement = 0
        self.timeOfReduction = 0
        self.rateOfReduction = 1


class Performance():
    def __init__(self, deterioration_model, maintenance_actions):
        self.deterioration_model = deterioration_model
        self.actions_schedule = {}
        self.action_effects = ActionEffect.set_action_effects(
            deterioration_model._number_of_states,
            maintenance_actions)
        
        self.list_of_possible_ICs = np.linspace(start = self.deterioration_model.best_IC,
                                                stop = self.deterioration_model.worst_IC,
                                                num = self.deterioration_model._number_of_states,
                                                dtype = int)
        self.Q = self.deterioration_model.intensity_matrix
        self.standard_transition_matrix = expm(self.Q)

        self._number_of_process = 1

    def get_action(self, time) -> List:
        return self.actions_schedule.get(str(time), None)

    def _set_actions_schedule(self, actions_schedule: Dict):
        self.actions_schedule = actions_schedule

    def get_reduction_factor(self,
                             sample: dict,
                             time: int):
        reduction_factor = 1
        
        for t, s in sample.items():
            if time < t:
                continue
            time_difference = time - t
            if s.timeOfReduction - time_difference > 0 and s.rateOfReduction < reduction_factor:
                reduction_factor = s.rateOfReduction
                self.last_intervention['reduction'] = reduction_factor
                
            if s.timeOfDelay - time_difference > 0:
                # there is suppression (P - is identity matrix)
                reduction_factor = 0
                self.last_intervention['reduction'] = reduction_factor
                break
        
        return reduction_factor

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

    def _get_next_IC(self,
                     current_state: int,
                     interventions: dict,
                     time: int,
                     action):
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
        # ação corretiva
        if interventions.get(time, None):
            if interventions[time].improvement:
                return self.get_improved_IC(current_state,
                                            interventions[time].improvement)

        # compute rate of reduction
        reduction_factor = self.get_reduction_factor(
            interventions, time)

        if(self.last_intervention['time']+self.last_intervention['duration'] <= time):
            self.last_intervention['reduction'] = 1
        # print('---')
        # print('time: ', time)
        # print(self.last_intervention['reduction'], '||', reduction_factor)
        # print('---')
        reduction_factor = self.last_intervention['reduction']

        if reduction_factor == 0:
            return current_state

        # compute transition matrix
        transition_matrix = self.standard_transition_matrix
        if reduction_factor != 1:
            intensity_matrix_reduced = self.Q * reduction_factor
            transition_matrix = expm(intensity_matrix_reduced)

        # calculate next state
        next_IC = self._choose_randomly_the_next_IC(current_state,
                                                    transition_matrix)
        return next_IC
        # return max(self.deterioration_model.get_next_IC(current_state),
        #            current_state)

    def set_interventions_effect(self, intervention, action, IC):
        IC_index = int(IC)
        
        intervention.timeOfDelay = action.get_time_of_delay(IC_index)
        intervention.improvement = action.get_improvement(IC_index)
        intervention.timeOfReduction = action.get_time_of_reduction(IC_index)
        intervention.rateOfReduction = action.get_reduction_rate(IC_index)
        
        self.last_intervention['duration'] = max(intervention.timeOfDelay,
                                                 intervention.timeOfReduction)
        self.last_intervention['reduction'] = 1

   
    def predict_MC(self,
                   time_horizon,
                   initial_IC):
        #Time horizon for years [0 + time_horizon]
        time_horizon += 1

        asset_condition = np.empty(time_horizon, dtype=int)
        asset_condition[0] = initial_IC
        
        # I still need to get rid off the ´Sample´ class
        #interventions = [Sample() for _ in range(time_horizon)]
        interventions = {int(year): Sample() for year, action in self.actions_schedule.items()}
        
        self.last_intervention = {"time": 0,
                                  "duration": 0,
                                  "reduction": 1}
        
        ## Cache the method reference for better performance
        for time in range(1, time_horizon):
            action = self.get_action(time)
            if self.action_effects.get(action, None):
                self.last_intervention['time'] = time
                self.set_interventions_effect(interventions[time],
                                              self.action_effects[action],
                                              asset_condition[time-1])
            
            asset_condition[time] = self._get_next_IC(asset_condition[time-1],
                                                      interventions,
                                                      time,
                                                      action)
        return asset_condition

    def get_IC_over_time(self,
                         time_horizon: int,
                         initial_IC: int = None,
                         actions_schedule: Dict = {},
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
        if not initial_IC: #if initial_IC is None:
            initial_IC = self.deterioration_model.best_IC
        self._set_actions_schedule(actions_schedule)
        
        if self._number_of_process == 1:
            samples = [self.predict_MC(time_horizon,initial_IC) for _ in range(number_of_samples)]
            return np.mean(samples, axis=0)  # Mean pear year
            
        
        chunk_size = max(number_of_samples // self._number_of_process, 1)
        chunks = [(time_horizon, initial_IC)] * number_of_samples
        with Pool(processes=self._number_of_process) as p:
            # pool_results = p.starmap(self.predict_MC, chunks, chunksize=chunk_size)
            # samples = np.array(pool_results)
            pool_results = [p.apply_async(self.predict_MC,
                                          (
                                              time_horizon,
                                              initial_IC)
                                          )
                            for _ in range(number_of_samples)]
            samples = [result.get() for result in pool_results]

        return np.mean(samples, axis=0)  # Mean pear year
from os import getpid