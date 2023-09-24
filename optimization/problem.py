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

class ASFiNAGProblem(MaintenanceSchedulingProblem):
    """
    Maintenance scheduling optimization problem.
    """
    
    def extract_indicator(indicator, actions):
        final_data = []

        for data in actions:
            # Extract the indicator if it exists
            pi_data = data.get(indicator)
            if pi_data is None:
                continue
            
            # Create the new dictionary with the desired structure
            extracted_data = {
                "name": data.get("name"),
            }
            
            if "time_of_reduction" in pi_data:
                extracted_data["time_of_reduction"] = pi_data["time_of_reduction"]
            
            if "reduction_rate" in pi_data:
                extracted_data["reduction_rate"] = pi_data["reduction_rate"]
            
            if "improvement" in pi_data:
                extracted_data["improvement"] = pi_data["improvement"]
            
            extracted_data["cost"] = data.get("cost")
            
            final_data.append(extracted_data)
        return final_data
    
    def __init__(self, performance_models, actions, time_horizon, **kwargs):
        """
        Initialize the problem.

        Args:
            markov: Markov deterioration model 
            maintenance_file: JSON file with maintenance actions
            time_horizon: Planning horizon
        """
        self.time_horizon = time_horizon
        self.performance_models = performance_models
        self.number_of_samples = 10
        
        self.discount_rate = 0.01
        
        # Constrains
        self.max_budget = 15#np.inf
        self.max_indicator = 5#np.inf
        
        # Index actions by name using a dictionary
        self.actions = dict((d['name'], dict(d, index=index)) 
                             for (index, d) in enumerate(actions))
        
        num_actions_database = len(self.actions)
        max_actions = 5 # Total number of actions in the analysis
        n_var = 5 * 2 # 2 represents "time" and "action"
        
        xl = [0, 0] * max_actions
        xu = [self.time_horizon, num_actions_database-1] * max_actions
        
        Problem.__init__(self,
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
        f1 = self._calc_global_area_under_curve(performances)

        # Minimize cost
        f2 = self._calc_budget(x)
        
        out["F"] = [f1, f2]

        #Constraints
        #Maximum number of interventions
        g1 = f2 - self.max_budget
        
        #Maximum indicator
        g2 = self._calc_max_global_indicator(performances) - self.max_indicator
        
        out["G"] = [g1, g2]
        
    def _evaluate_performance(self, xs):
        """
        Calculate performance for population.
        """
        return np.array([self._get_performances(self._decode_solution(x)) for x in xs])
    
    def _calc_global_area_under_curve(self, xs):
        results = self._calc_area_under_curve(xs)
        return np.array([max(result.values()) for result in results])
    
    def _calc_area_under_curve(self, performances_list):
        """
        Calculate area under curve for population.
        """
        result_list = []
        for performances in performances_list:
            result_dict = {}
            for key, performance in performances.items():
                result_dict[key] = sum(performance)
            result_list.append(result_dict)
        return np.array(result_list)
    
    def _calc_max_global_indicator(self, performances):
        results = self._calc_max_indicator(performances)
        return np.array([max(result.values()) for result in results])
    
    def _calc_max_indicator(self, performances_list):
        """
        Calculate max indicator for population.
        """
        result_list = []
        for performances in performances_list:
            result_dict = {}
            for key, performance in performances.items():
                result_dict[key] = max(performance)
            result_list.append(result_dict)
        return np.array(result_list)
        
        #return np.array([max(performance) for performance in performances])
    
    def _get_performances(self, actions_schedule):
        """
        Get performance indicator over time.
        """
        result = {}
        for key, performance_model in self.performance_models.items():
            result[key] = performance_model.get_IC_over_time(
                time_horizon=self.time_horizon,
                actions_schedule=actions_schedule,
                number_of_samples=self.number_of_samples
            )
        return result

    def _calc_budget(self, xs):
        """
        Calculate maintenance budget for population. 
        """
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
        # """
        return (self.actions[action]['cost']
                / (1.0 + self.discount_rate)**float(year))

    def _decode_solution(self, binary):
        """
        Decode binary solution to actions schedule.
        """
        binary_year = binary.reshape(-1, 2)
        actions = {}
        
        for year, action in binary_year:
            if action !=0 : #Excluding DoNothing
                actions[str(year)] = list(self.actions)[action-1]
        
        return actions
        
    