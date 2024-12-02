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

        hazard_data = [
            {
                "name": "Minor Damage",
                "probability": 0.161281,
                "degradation":   {
                    "1": [1, 1, 1],
                    "2": [1, 1, 1],
                    "3": [1, 1, 1],
                    "4": [1, 1, 1],
                }
            },
            {
                "name": "Moderate Damage",
                "probability": 0.068461,
                "degradation":   {
                    "1": [2, 2, 2],
                    "2": [2, 2, 2],
                    "3": [2, 2, 2],
                    "4": [2, 2, 2],
                }
            },{
                "name": "Severe Damage",
                "probability": 0.022321,
                "degradation":   {
                    "1": [3, 3, 3],
                    "2": [3, 3, 3],
                    "3": [3, 3, 3],
                    "4": [3, 3, 3],
                }
            },{
                "name": "Collapse",
                "probability": 0.000429,
                "degradation":   {
                    "1": [4, 4, 4],
                    "2": [4, 4, 4],
                    "3": [4, 4, 4],
                    "4": [4, 4, 4],
                }
            },
        ]

        
        model = Performance(markov,
                            actions,
                            hazard_data)
        
        time_horizon = 20
        
        problem = MaintenanceSchedulingProblem(model, time_horizon, max_actions=10)
        problem.performance_model._number_of_process = 1
        
        self.optimization = Multi_objective_optimization()
        self.optimization.verbose = False
        self.optimization.set_problem(problem)
        
        self.optimization._set_algorithm({"name": "NSGA2", "pop_size": 5, "eliminate_duplicates": True})
        self.optimization._set_termination({'name':'n_gen', 'n_max_gen':3})
        
    def test_minimize(self):
        np.random.seed(1)
        random.seed(1)
        
        res = self.optimization.minimize()
        
        sort = np.argsort(res.F.T)[1]
        
        performance = res.F.T[0][sort]
        cost = res.F.T[1][sort]
        best_action = self.optimization.problem._decode_solution(res.X[sort][-1])
        
        self.assertAlmostEqual(performance[0], 90.5, places=3)
        self.assertAlmostEqual(performance[-1], 63.1, places=3)
        
        self.assertAlmostEqual(cost[0], 6.50725405, places=3)
        self.assertAlmostEqual(cost[-1], 15.42969574, places=5)
        
        action = {'18': 'action_1', '10': 'action_2', '12': 'action_1'}
        
        self.assertEqual(action, best_action)

    def test_minimize_with_hazard_schedule(self):
        hazards_schedule = {1: 'Collapse'}
        
        self.optimization.problem.hazards_schedule = hazards_schedule
        np.random.seed(1)
        random.seed(1)

        res = self.optimization.minimize()
        
        sort = np.argsort(res.F.T)[1]
        
        performance = res.F.T[0][sort]
        cost = res.F.T[1][sort]
        best_action = self.optimization.problem._decode_solution(res.X[sort][-1])
        
        self.assertAlmostEqual(performance[0], 101, places=3)
        self.assertAlmostEqual(performance[-1], 48.1, places=3)
        
        self.assertAlmostEqual(cost[0], 0, places=3)
        self.assertAlmostEqual(cost[-1], 35.4427383632713, places=5)
        
        action = {'15': 'action_1', '0': 'action_1', '16': 'action_1', '9': 'action_1', '2': 'action_2', '13': 'action_1', '11': 'action_2'}
        
        self.assertEqual(action, best_action)
   
    def test_max_indicator_constrain(self):
        max_indicator = 3
        self.optimization.problem.max_indicator = max_indicator
        self.optimization._set_algorithm({"name": "NSGA2", "pop_size": 20, "eliminate_duplicates": True})
        self.optimization._set_termination({'name':'n_gen', 'n_max_gen':20})
        
        np.random.seed(1)
        random.seed(1)
        res = self.optimization.minimize()

        sort = np.argsort(res.F.T)[1]
        
        self.assertAlmostEqual(res.F.T[0][sort][-1], 49.4, places=3)
        best_action = self.optimization.problem._decode_solution(res.X[sort][-1])
        
        performance = self.optimization.problem._get_performance(best_action)
        
        self.assertTrue(max(performance) <= max_indicator+.5)
    


     

    

        

        
   
if __name__ == '__main__':
    unittest.main()