import unittest
import warnings
import random
import numpy as np

from ams.prediction.markov import MarkovContinous

from ams.performance.performance import Performance

from ams.optimization.problem import MaintenanceSchedulingProblem, MultiIndicatorProblem
from ams.optimization.multi_objective_optimization import Multi_objective_optimization
   
class Test_MaintenanceSchedulingProblem_Optimization(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning, module='pymoo')
        markov = MarkovContinous(worst_IC=5, best_IC=1)
        markov.theta = [0.508275, 0.410478, 0.281638, 0.247331, 0.      ]
        
        actions = [{"name": 'action_1',
                   "time_of_reduction": {
                                2: [5, 5, 5],
                                3: [5, 5, 5]
                            },
                   "reduction_rate":    {
                            2: [0.1, 0.1, 0.1],
                            3: [0.1, 0.1, 0.1]
                         },
                   "cost": 3.70
                   },
                   {"name": 'action_2',
                   "improvement": {
                                2: [1, 1, 1],
                                3: [2, 2, 2],
                                4: [3, 3, 3],
                                5: [4, 4, 4]
                            },
                   "cost": 10
                   },
        ]
        
        time_horizon = 20
        
        model = Performance(markov, actions)
        
        problem = MaintenanceSchedulingProblem(model, time_horizon)
        problem.performance_model._number_of_process = 1
        
        self.optimization = Multi_objective_optimization()
        self.optimization.verbose = False
        self.optimization.set_problem(problem)
        
        self.optimization._set_algorithm({"name": "NSGA2", "pop_size": 20, "eliminate_duplicates": True})
        self.optimization._set_termination({'name':'n_gen', 'n_max_gen':3})
        
    def test_minimize(self):
        np.random.seed(1)
        random.seed(1)
        
        res = self.optimization.minimize()
        
        sort = np.argsort(res.F.T)[1]
        
        performance = res.F.T[0][sort]
        cost = res.F.T[1][sort]
        best_action = self.optimization.problem._decode_solution(res.X[sort][-1])
        
        self.assertAlmostEqual(performance[0], 81.8, places=3)
        self.assertAlmostEqual(performance[-1], 42, places=3)
        
        self.assertAlmostEqual(cost[0], 0, places=3)
        self.assertAlmostEqual(cost[-1], 30.820983863017375, places=5)
        
        action = {'12': 'action_2', '3': 'action_2', '2': 'action_1', '15': 'action_2'}
        
        self.assertEqual(action, best_action)
    
    def test_budget_constrain(self):
        max_budget = 5
        self.optimization.problem.max_budget = max_budget
        
        np.random.seed(1)
        random.seed(1)
        res = self.optimization.minimize()        
        
        cost = res.F.T[1]
        
        self.assertTrue(max(cost) < max_budget)
    
    def test_max_indicator_constrain(self):
        max_indicator = 3
        self.optimization.problem.max_indicator = max_indicator
        self.optimization._set_termination({'name':'n_gen', 'n_max_gen':5})
        
        np.random.seed(1)
        random.seed(1)
        res = self.optimization.minimize()  
        
        actions_schedule = self.optimization.problem._decode_solution(res.X[-1])
        performance = self.optimization.problem._get_performance(actions_schedule)
        
        self.assertTrue(max(performance) <= max_indicator+1)
    
    def test_budget_indicator_constrain(self):
        max_budget = 15
        max_indicator = 3
        
        self.optimization.problem.max_budget = max_budget
        self.optimization.problem.max_indicator = max_indicator
        
        np.random.seed(2)
        random.seed(2)
        self.optimization._set_termination({'name':'n_gen', 'n_max_gen':7})
        res = self.optimization.minimize()
        
        self.assertTrue(res.G[0][1] <= 0.3)
        cost = res.F.T[1]
        self.assertTrue(max(cost) < max_budget)

class Test_MultiIndicatorProblem_Optimization(unittest.TestCase):
    def setUp(self):
        # Thetas
        PI_B = [0.0186, 0.0256, 0.0113, 0.0420]
        PI_CR = [0.0736, 0.1178, 0.1777, 0.3542]
        PI_E = [0.0671, 0.0390, 0.0489, 0.0743]
        PI_F = [0.1773, 0.2108, 0.1071, 0.0765]
        PI_R = [0.1084, 0.0395, 0.0443, 0.0378]

        #Mapping thetas and indicators
        thetas = {'Bearing_Capacity': PI_B,
                  'Cracking':PI_CR,
                  'Longitudinal_Evenness': PI_E,
                  'Skid_Resistance': PI_F,
                  'Transverse_Evenness': PI_R}

        # Set actions database
        actions = [{"name": 'action_1',
                    "Bearing_Capacity": {
                        "time_of_reduction": {
                                2: [5, 5, 5],
                                3: [5, 5, 5]
                        },
                       "reduction_rate":    {
                                2: [0.1, 0.1, 0.1],
                                3: [0.1, 0.1, 0.1]
                        }
                    },
                    "Cracking": {
                        "improvement": {
                                2: [1, 1, 1],
                                3: [2, 2, 2],
                                4: [3, 3, 3],
                                5: [4, 4, 4]
                        }
                    },
                    "Skid_Resistance": {
                        "improvement": {
                            2: [1, 1, 1],
                            3: [2, 2, 2],
                            4: [3, 3, 3],
                            5: [4, 4, 4]
                        }
                    },
                   "cost": 3.70
                   },
                   {"name": 'action_2',
                    "Transverse_Evenness": {
                        "improvement": {
                            2: [1, 1, 1],
                            3: [2, 2, 2],
                            4: [3, 3, 3],
                            5: [4, 4, 4]
                        }
                    },
                    "Longitudinal_Evenness": {
                        "improvement":  {
                            2:[1, 1, 1], 
                            3:[2, 2, 2], 
                            4:[2, 2, 2]
                        }
                    },
                    "Skid_Resistance": {
                        "improvement": {
                            2: [1, 1, 1],
                            3: [2, 2, 2],
                            4: [3, 3, 3],
                            5: [4, 4, 4]
                        }
                    },
                   "cost": 3
                   },
        ]

        # Create one performance model for each indicator
        performance_models = {}
        for key, theta in thetas.items():
            markov = MarkovContinous(worst_IC=5, best_IC=1)
            markov.theta = theta
            filtered_actions = MultiIndicatorProblem.extract_indicator(key, actions)
            performance_models[key] = Performance(markov, filtered_actions)

        problem = MultiIndicatorProblem(performance_models, time_horizon=20)
        
        self.optimization = Multi_objective_optimization()
        self.optimization.verbose = False
        self.optimization.set_problem(problem)
        
        self.optimization._set_algorithm({"name": "NSGA2", "pop_size": 20, "eliminate_duplicates": True})
        self.optimization._set_termination({'name':'n_gen', 'n_max_gen':3})
        
    def test_minimize(self):
        np.random.seed(1)
        random.seed(1)

        res = self.optimization.minimize()

        sort = np.argsort(res.F.T)[1]

        performance = res.F.T[0][sort]
        cost = res.F.T[1][sort]
        best_action = self.optimization.problem._decode_solution(res.X[sort][-1])

        self.assertAlmostEqual(performance[0], 52.3, places=3)
        self.assertAlmostEqual(performance[-1], 29.9, places=3)

        self.assertAlmostEqual(cost[0], 0, places=3)
        self.assertAlmostEqual(cost[-1], 12.36930824695356, places=5)

        action = {'9': 'action_2', '3': 'action_1', '7': 'action_1', '15': 'action_2'}
        
        self.assertEqual(action, best_action)
        
    def test_budget_constrain(self):
        max_budget = 3
        self.optimization.problem.max_budget = max_budget

        np.random.seed(1)
        random.seed(1)
        res = self.optimization.minimize()        

        cost = res.F.T[1]
        
        self.assertTrue(max(cost) < max_budget)
        
    def test_global_max_indicator_constraint(self):
        max_global_indicator = 3
        self.optimization.problem.max_global_indicator = max_global_indicator

        np.random.seed(1)
        random.seed(1)
        res = self.optimization.minimize()  
        
        most_expensive_solution = res.X[-1]
        actions_schedule = self.optimization.problem._decode_solution(most_expensive_solution)
        performance = self.optimization.problem._get_performances(actions_schedule)
        global_indicator = self.optimization.problem._calc_max_global_indicator([performance])
        self.assertTrue(global_indicator <= max_global_indicator)
        
        cheapest_solution = res.X[0]
        actions_schedule = self.optimization.problem._decode_solution(cheapest_solution)
        performance = self.optimization.problem._get_performances(actions_schedule)
        global_indicator = self.optimization.problem._calc_max_global_indicator([performance])
        self.assertTrue(global_indicator <= max_global_indicator)
        
    def test_single_indicator_constraint(self):
        max_indicators = {'Cracking': 2}
        
        self.optimization.problem.single_indicators_constraint = max_indicators
        
        np.random.seed(1)
        random.seed(1)
        self.optimization._set_termination({'name':'n_gen', 'n_max_gen':10})
        res = self.optimization.minimize()
        
        sort = np.argsort(res.F.T)[1]
        
        most_expensive_solution = res.X[sort][-1]
        actions_schedule = self.optimization.problem._decode_solution(most_expensive_solution)
        performance = self.optimization.problem._get_performances(actions_schedule)
        indicators_diff = self.optimization.problem._calc_max_indicators([performance])
        
        self.assertTrue(max(indicators_diff)[0] <= 0.101)
    
    def test_single_indicators_constraint(self):
        max_indicators = {'Bearing_Capacity': 2,
                          'Cracking': 2,
                          'Longitudinal_Evenness': 2,
                          'Skid_Resistance': 2,
                          'Transverse_Evenness': 2}
        
        self.optimization.problem.single_indicators_constraint = max_indicators
        
        np.random.seed(1)
        random.seed(1)
        self.optimization._set_termination({'name':'n_gen', 'n_max_gen':20})
        res = self.optimization.minimize()
        
        sort = np.argsort(res.F.T)[1]
        
        most_expensive_solution = res.X[sort][-1]
        actions_schedule = self.optimization.problem._decode_solution(most_expensive_solution)
        performance = self.optimization.problem._get_performances(actions_schedule)
        indicators_diff = self.optimization.problem._calc_max_indicators([performance])
        
        self.assertTrue(max(indicators_diff)[0] <= 0.21)
        
    def test_budget_indicator_constrain(self):
        max_budget = 3.5
        max_global_indicator = 3

        self.optimization.problem.max_budget = max_budget
        self.optimization.problem.max_global_indicator = max_global_indicator

        np.random.seed(2)
        random.seed(2)
        res = self.optimization.minimize()
        
        ## Indicator
        cheapest_solution = res.X[0]
        actions_schedule = self.optimization.problem._decode_solution(cheapest_solution)
        performance = self.optimization.problem._get_performances(actions_schedule)
        max_global_indicator = self.optimization.problem._calc_max_global_indicator([performance])        
        self.assertTrue(max_global_indicator <= max_global_indicator)
        
        ## Cost
        cost = res.F.T[1]
        self.assertTrue(max(cost) < max_budget)
        
   
if __name__ == '__main__':
    unittest.main()