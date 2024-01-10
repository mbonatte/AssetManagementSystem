"""
Created on Sep 21, 2022.

@author: MauricioBonatte
@e-mail: mbonatte@ymail.com
"""

from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem

from ams.prediction.markov import MarkovContinous
from ams.performance.maintenance import ActionEffect
from ams.performance.performance import Performance

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
        self.number_of_samples = 10
        
        self.discount_rate = 0.01
        
        # Constrains
        self.max_budget = np.inf
        self.max_indicator = np.inf
        
        num_actions_database = len(self.actions)
        max_actions = 5 # Total number of actions in the analysis
        n_var = 5 * 2 # 2 represents "time" and "action"
        
        xl = [0, 0] * max_actions
        xu = [self.time_horizon, num_actions_database-1] * max_actions
        
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=2,
            xl=xl,
            xu=xu,
            vtype=int,
            **kwargs
            )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the objective functions.
        """
        
        performances = self._evaluate_performance(x)
        
        # Objectives
        
        # Minimize performance indicator
        f1 = self._calc_area_under_curve(performances)

        # Minimize cost
        f2 = self._calc_budget(x)
        
        out["F"] = [f1, f2]

        #Constraints
        #Maximum number of interventions
        g1 = f2 - self.max_budget
        
        #Maximum indicator
        g2 = self._calc_max_indicator(performances) - self.max_indicator
        
        out["G"] = [g1, g2]
        
    def _evaluate_performance(self, xs):
        """
        Calculate performance for population.
        """
        return np.array([self._get_performance(self._decode_solution(x)) for x in xs])
    
    def _calc_area_under_curve(self, performances):
        """
        Calculate area under curve for population.
        """
        return np.array([sum(performance) for performance in performances])
    
    def _calc_max_indicator(self, performances):
        """
        Calculate max indicator for population.
        """
        return np.array([max(performance) for performance in performances])
    
    def _get_performance(self, actions_schedule):
        """
        Get performance indicator over time.
        """
        return self.performance_model.get_IC_over_time(
            time_horizon=self.time_horizon,
            actions_schedule=actions_schedule,
            number_of_samples=self.number_of_samples
        )

    def _calc_budget(self, xs):
        """
        Calculate maintenance budget for population. 
        """
        #return self._get_budget(self._decode_solution(xs))
        return np.array([self._get_budget(self._decode_solution(x)) for x in xs])

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
        return (self.actions[action].cost
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
        model._number_of_process = 1
        return model

    def _decode_solution(self, binary):
        """
        Decode binary solution to actions schedule.
        """
        binary_year = binary.reshape(-1, 2)
        actions = {}
        for year, action in binary_year:
            if action !=0 :
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

class NetworkProblem(Problem):
    """
    Network optimization problem.
    
    In the network optimization, each element of the population represents the
    index of the solution in the Pareto set obtained by section optimization,
    with position in the population referring to the corresponding pavement section 
    in the road network. For instance, the population [3 5 2 7 6 1 2 5 4 9] encodes 
    that the third solution from the Pareto set must be used for the first section, 
    whereas for the second section the fifth solution must be implemented, and so on.
    """

    def __init__(self, section_optimization, **kwargs):
        """
        Initialize the network problem.

        Args:
            section_optimization: dictionary with results from the section optimization
        """
        self.section_optimization = section_optimization
        
        n_sections = len(self.section_optimization) # Number of sections in the analysis
        
        xl = [0] * n_sections
        xu = [len(n['Performance'])-1 for n in self.section_optimization.values()]
        
        super().__init__(
            n_var=n_sections,
            n_obj=2,
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
            vtype=int,
            **kwargs
            )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the objective functions.
        """
        
        # Objectives
        
        # Minimize network performance indicator
        f1 = self._calc_network_performance_indicator_pop(x)

        # Minimize cost
        f2 = self._calc_network_budget_pop(x)
        
        out["F"] = [f1, f2]
        
        
    def _calc_network_performance_indicator_pop(self, xs):
        mean_performances = []
        
        for x in xs:
            mean_performances.append(self._calc_network_performance_indicator(x))
        
        return mean_performances
    
    def _calc_network_performance_indicator(self, xs):
        performances = []
        
        for x, section in zip(xs, self.section_optimization.values()):
            performances.append(section['Performance'][x])
        
        return np.mean(performances)
    
    def _calc_network_budget_pop(self, xs):
        sum_costs = []
        
        for x in xs:
            sum_costs.append(self._calc_network_budget(x))
        
        return sum_costs
    
    def _calc_network_budget(self, xs):
        costs = []
        
        for x, section in zip(xs, self.section_optimization.values()):
            costs.append(section['Cost'][x])
        
        return np.sum(costs)
