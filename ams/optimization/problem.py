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

    def __init__(self, performance_model, time_horizon, initial_IC=None, max_actions=5, discount_rate=0.01, number_of_samples=10, **kwargs):
        """
        Initialize the maintenance scheduling problem.

        Args:
            performance_model (Performance): Performance deterioration/maintenance model.
            time_horizon (int): Planning horizon.
            initial_IC : The initial condition index.
            max_actions (int): Maximum number of actions.
            discount_rate (float): Discount rate for cost calculation.
            number_of_samples (int): Number of samples for performance calculation.
        """
        self.time_horizon = time_horizon
        self.performance_model = performance_model
        self.initial_IC = initial_IC
        self.actions = self.performance_model.action_effects
        
        self.discount_rate = discount_rate
        self.number_of_samples = number_of_samples
        
        # Constrains
        n_ieq_constr = 2
        self.max_budget = np.inf
        self.max_indicator = self.performance_model.deterioration_model.worst_IC
        
        # Total number of actions in the analysis
        num_actions_database = len(self.actions)
        n_var = max_actions * 2 # 2 represents the binary ["time" and "action"]
        
        xl = max_actions * [0, 0]
        xu = max_actions * [self.time_horizon, num_actions_database - 1]
        
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=n_ieq_constr,
            xl=xl,
            xu=xu,
            vtype=int,
            **kwargs
            )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the objective functions and constraints for given solutions.

        Args:
            x: Solutions to evaluate.
            out: Dictionary to store the output results.
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
        Calculate performance for each solution in the population.

        Args:
            xs (list): List of solutions to evaluate.

        Returns:
            np.array: Array of performance values for each solution.
        """
        return np.array([self._get_performance(self._decode_solution(x)) for x in xs])
    
    def _calc_area_under_curve(self, performances):
        """
        Calculate the area under the curve for each performance prediction.

        Args:
            performances (np.array): Array of performance values.

        Returns:
            np.array: Area under the curve for each performance prediction.
        """
        return np.sum(performances, axis=1)
    
    def _calc_max_indicator(self, performances):
        """
        Calculate the maximum indicator value for a set of performance values.

        Args:
            performances (np.array): Array of performance values.

        Returns:
            np.array: Maximum indicator value for each set of performance values.
        """
        return np.max(performances, axis=1)
    
    def _get_performance(self, action_schedule):
        """
        Get the performance indicator over time for a given action schedule.

        Args:
            action_schedule (dict): Schedule of actions.

        Returns:
            list: Performance indicator values over time.
        """
        return self.performance_model.get_IC_over_time(
            time_horizon = self.time_horizon,
            initial_IC = self.initial_IC,
            actions_schedule = action_schedule,
            number_of_samples = self.number_of_samples
        )

    def _calc_budget(self, xs):
        """
        Calculate the maintenance budget for each solution in the population.

        Args:
            xs (list): List of solutions to evaluate.

        Returns:
            np.array: Maintenance budget for each solution.
        """
        return np.array([self._get_budget(self._decode_solution(x)) for x in xs])

    def _get_budget(self, action_schedule):
        """
        Calculate the total discounted budget for a given action schedule.

        Args:
            action_schedule (dict): Schedule of actions.

        Returns:
            float: Total discounted budget.
        """
        budget = sum(self._get_action_cost(int(year), action) for year, action in action_schedule.items())
        return budget
        
    def _get_action_cost(self, year, action):
        """
        Calculate the cost of an action, discounted to the present value.

        Args:
            year (int): Year of the action.
            action (str): Action to be taken.

        Returns:
            float: Discounted cost of the action.
        """
        return self.actions[action].cost / ((1.0 + self.discount_rate) ** year)
    
    def _get_number_of_interventions(self, solution):
        """
        Count the number of maintenance interventions in a solution.

        Args:
            solution (list): Solution to evaluate.

        Returns:
            int: Number of maintenance interventions.
        """
        action_schedule = self._decode_solution(solution)
        return sum(1 for action in action_schedule.values() if action != 'DoNothing')

    def _decode_solution(self, solution):
        """
        Decode a binary solution into a schedule of actions.

        Args:
            solution (list): Binary solution to decode.

        Returns:
            dict: Schedule of actions derived from the solution.
        """
        actions_schedule = {}
        for year, action in solution.reshape(-1, 2):
            if action != 0:  # Excluding 'DoNothing' action
                actions_schedule[str(year)] = list(self.actions)[action]
        return dict(sorted(actions_schedule.items()))

class MultiIndicatorProblem(MaintenanceSchedulingProblem):
    """
    Maintenance scheduling optimization problem for multiple performance indicators.
    """
    
    @staticmethod
    def extract_indicator(indicator, actions):
        """
        Extracts specified indicators from a collection of actions.

        Args:
            indicator (str): The indicator to be extracted.
            actions (list): List of action dictionaries containing the indicators.

        Returns:
            list: List of dictionaries with extracted indicator data.
        """
        final_data = []

        for data in actions:
            pi_data = data.get(indicator)
            if pi_data is None:
                continue
            
            # Create the new dictionary with the desired structure
            extracted_data = {
                "name": data.get("name"),
                "cost": data.get("cost"),
                **{key: pi_data[key] for key in ["time_of_reduction", "reduction_rate", "improvement"] if key in pi_data}
            }            
            
            final_data.append(extracted_data)
        
        return final_data
    
    def __init__(self, performance_models, time_horizon, initial_ICs={}, max_actions=5, discount_rate=0.01, number_of_samples=10, **kwargs):
        """
        Initialize the multi-indicator maintenance scheduling problem.

        Args:
            performance_models (dict): Dictionary of performance models.
            time_horizon (int): Planning horizon.
            discount_rate (float): Discount rate for cost calculation.
            number_of_samples (int): Number of samples for performance calculation.
        """
        self.time_horizon = time_horizon
        self.performance_models = performance_models
        self.initial_ICs = initial_ICs
        
        self.actions = {name: action for model in performance_models.values() for name, action in model.action_effects.items()}
        
        self.number_of_samples = number_of_samples
        
        self.discount_rate = discount_rate
        
        # Constrains
        self.max_budget = np.inf
        self.max_indicator = np.inf
        
        num_actions_database = len(self.actions)
        n_var = max_actions * 2  # "time" and "action"
        
        xl = max_actions * [0, 0]
        xu = max_actions * [self.time_horizon, num_actions_database - 1]
        
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
        Evaluate the objective functions and constraints for the given solutions.

        Args:
            x: Solutions to evaluate.
            out: Dictionary to store the output results.
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
        Calculate performance for each solution in the population.

        Args:
            xs (list): List of solutions to evaluate.

        Returns:
            np.array: Array of performance values for each solution.
        """
        return np.array([self._get_performances(self._decode_solution(x)) for x in xs])
    
    def _calc_global_area_under_curve(self, performances):
        """
        Calculate the global area under the curve for a set of performance values.

        Args:
            performances (list): List of {indicator: performance} dictionaries.
        Returns:
            np.array: Global area under the curve for each set of performance values.
        """
        results = self._calc_area_under_curve(performances)
        return np.array([max(result.values()) for result in results])
        
    def _calc_max_global_indicator(self, performances):
        """
        Calculate the maximum global indicator value for a set of performance values.

        Args:
            performances (list): List of {indicator: performance} dictionaries.

        Returns:
            np.array: Maximum global indicator value for each set of performance values.
        """
        results = self._calc_max_indicator(performances)
        return np.array([max(result.values()) for result in results])
    
    def _calc_area_under_curve(self, performances_list):
        """
        Calculate area under the curve for each indicator in the population.

        Args:
            performances_list (list): List of {indicator: performance} dictionaries.

        Returns:
            np.array: Area under the curve for each indicator/solution.
        """
        result_list = []
        for performances in performances_list:
            result_dict = {}
            for key, performance in performances.items():
                result_dict[key] = sum(performance)
            result_list.append(result_dict)
        return np.array(result_list)
        
        return np.array([sum(performance) for performance in performances_list])
    
    
    def _calc_max_indicator(self, performances_list):
        """
        Calculate the maximum performance indicator for each indicator in the population.

        Args:
            performances_list (list): List of {indicator: performance} dictionaries.

        Returns:
            np.array: Maximum indicator for each indicator/solution.
        """
        result_list = []
        for performances in performances_list:
            result_dict = {}
            for key, performance in performances.items():
                result_dict[key] = max(performance)
            result_list.append(result_dict)
        return np.array(result_list)
    
    def _get_performances(self, action_schedule):
        """
        Get performance indicators over time for a given action schedule.

        Args:
            action_schedule (dict): Schedule of actions.

        Returns:
            dict: Performance indicators for each model.
        """
        return {key: model.get_IC_over_time(
                    time_horizon=self.time_horizon,
                    initial_IC=self.initial_ICs.get(key, None),
                    actions_schedule=action_schedule,
                    number_of_samples=self.number_of_samples)
                for key, model in self.performance_models.items()}

class NetworkProblem(Problem):
    """
    Network optimization problem.
    
    In the network optimization, each element of the population represents the
    index of the solution in the Pareto set obtained by asset optimization,
    with position in the population referring to the corresponding asset. 
    For instance, the population [3 5 2 7 6 1 2 5 4 9] encodes 
    that the third solution from the Pareto set must be used for the first asset, 
    whereas for the second asset the fifth solution must be implemented, and so on.
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
