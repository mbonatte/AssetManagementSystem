"""
Created on Sep 21, 2022.

@author: MauricioBonatte
@e-mail: mbonatte@ymail.com
"""

from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem

from prediction.markov import MarkovContinous
from maintenance.maintenance import ActionEffect
from maintenance.performance import Performance

import numpy as np
import json

class MaintenanceSchedulingProblem(Problem):
    """
    Maintenance scheduling optimization problem.
    """

    def __init__(self, markov, maintenance_file, time_horizon, **kwargs):
        """
        Initialize the problem.

        Args:
            markov: Markov deterioration model 
            maintenance_file: JSON file with maintenance actions
            time_horizon: Planning horizon
        """
        self.time_horizon = time_horizon
        self.performance_model = self._create_performance_model(markov, maintenance_file)
        self.actions = self._set_actions(maintenance_file)
        self.discount_rate = 0.01
        
        n_var = 5 * 2
        super().__init__(n_var=n_var,
                         n_obj=2,
                         # n_ieq_constr=1,
                         xl=[0, 0] * 5,
                         xu=[self.time_horizon, len(self.performance_model.action_effects)-1] * 5,
                         vtype=int,
                         **kwargs
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the objective functions.
        """
        #print(x)
        # Objectives
        
        # Minimize performance indicator
        f1 = self._calc_performance(x)

        # Minimize cost
        f2 = self._calc_budget(x)
        
        out["F"] = [f1, f2]

        # Constraints
        # g1 = list(map(self._get_number_of_interventions, x))

        # out["G"] = [g1]
    
    def _calc_performance(self, xs):
        """
        Calculate performance indicator for population.
        """
        #return sum(self._get_performance(self._decode_solution(xs)))
        return [sum(self._get_performance(self._decode_solution(x))) for x in xs]
    
    def _get_performance(self, actions_schedule):
        """
        Get performance indicator over time.
        """
        return self.performance_model.get_IC_over_time(
            time_horizon=self.time_horizon,
            actions_schedule=actions_schedule,
            number_of_samples=10
        )

    def _calc_budget(self, xs):
        """
        Calculate maintenance budget for population. 
        """
        #return self._get_budget(self._decode_solution(xs))
        return [self._get_budget(self._decode_solution(x)) for x in xs]

    def _get_budget(self, actions_schedule):
        """
        Calculate total discounted budget.
        """
        budget = 0
        for year, action in actions_schedule.items():
            budget += self._get_action_cost(year, action)
        return budget
        
    def _get_action_cost(self, year, action):
        """
        Get action cost discounted.
        """
        return (self.performance_model.action_effects[action].cost
                / (1.0 + self.discount_rate)**float(year))
    
    def _get_number_of_interventions(self, x):
        """
        Get number of maintenance interventions.
        """
        actions = self._decode_solution(x) 
        return len([a for a in actions.values() if a != 'DoNothing'])

    def _create_performance_model(self, markov, maintenance_file):
        """
        Create performance model.
        """
        model = Performance(markov, maintenance_file)
        model._number_of_process = 32
        return model

    def _decode_solution(self, binary):
        """
        Decode binary solution to actions schedule.
        """
        binary_year = binary.reshape(-1, 2)
        actions = {}
        for year, action in binary_year:
            actions[str(year)] = list(self.actions)[action]
        return dict(sorted(actions.items()))

    def _set_actions(self, maintenance_file):
        """
        Set maintenance actions.
        """
        actions = ActionEffect.set_action_effects(
            self.performance_model.deterioration_model._number_of_states,
            maintenance_file)
        return actions
