"""
Created on Sep 21, 2022.

@author: MauricioBonatte
@e-mail: mbonatte@ymail.com
"""

from pymoo.core.problem import Problem

from prediction.markov import MarkovContinous
from maintenance.maintenance import ActionEffect
from maintenance.performance import Performance

import numpy as np
import json


class MyProblem(Problem):

    def __init__(self, markov, maintenance_file, time_horizon):
        self.time_horizon = 0
        self.actions = {}
        self.performance_model = None
        self.set_performance_model(markov, maintenance_file)
        self.discount_rate = 0.01
        self.time_horizon = time_horizon
        self._set_actions(maintenance_file)
        n_var = 5 * 2
        super().__init__(n_var=n_var,
                         n_obj=2,
                         # n_ieq_constr=1,
                         xl=[0, 0] * 5,
                         xu=[self.time_horizon, len(self.performance_model.action_effects)-1] * 5,
                         vtype=int
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        # each objective function is supposed to be minimized

        # minimize performance indicator
        f1 = list(map(self.calc_performance, x))

        # minimize cost
        f2 = list(map(self.calc_budget, x))
        out["F"] = [f1, f2]

        # constrain the maximum number of interventions
        # g1 = list(map(self.get_number_of_interventions, x))

        # out["G"] = [g1]

    def calc_performance(self, x):
        actions_schedule = self.decode_binary_to_dict(x)
        PI_over_time = self.performance_model.get_IC_over_time(time_horizon=self.time_horizon,
                                                               actions_schedule=actions_schedule,
                                                               number_of_samples=10
                                                               )
        return sum(PI_over_time)

    def calc_budget(self, x):
        actions_schedule = self.decode_binary_to_dict(x)
        budget = 0
        for ano in actions_schedule:
            action = actions_schedule[ano]
            budget += (self.performance_model.action_effects[action].cost
                       / (1.0 + self.discount_rate)**float(ano))
        return budget

    def get_number_of_interventions(self, x):
        print(len(self.decode_binary_to_dict(x)) - 5)
        print(self.decode_binary_to_dict(x))
        print('-------------------------------------------')
        return len(self.decode_binary_to_dict(x)) - 5

    def set_performance_model(self, markov, maintenance_file):
        self.performance_model = Performance(markov, maintenance_file)
        self.performance_model._number_of_process = 32

    def decode_binary_to_dict(self, binary):
        binary_year = binary.reshape(-1, 2)
        actions = {}
        for (year, action) in binary_year:
            actions[str(year)] = list(self.actions)[action]
        return dict(sorted(actions.items()))

    def _set_actions(self, maintenance_file):
        self.actions = ActionEffect.set_action_effects(
            self.performance_model.deterioration_model._number_of_states,
            maintenance_file)
