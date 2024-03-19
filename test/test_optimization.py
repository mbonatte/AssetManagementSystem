import unittest
import random
import numpy as np

from ams.prediction.markov import MarkovContinous

from ams.performance.performance import Performance

from ams.optimization.problem import MaintenanceSchedulingProblem, MultiIndicatorProblem
from ams.optimization.multi_objective_optimization import Multi_objective_optimization
   
class TestMulti_objective_optimization(unittest.TestCase):
    def setUp(self):
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
        
        self.assertAlmostEqual(performance[0], 74.9, places=3)
        self.assertAlmostEqual(performance[-1], 43.6, places=3)
        
        self.assertAlmostEqual(cost[0], 0, places=3)
        self.assertAlmostEqual(cost[-1], 28.18936, places=5)
        
        action =  {'13': 'action_1', '15': 'action_1', '18': 'action_1', '5': 'action_2', '9': 'action_2'}
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
        self.assertTrue(max(performance) <= max_indicator)
    
    def test_budget_indicator_constrain(self):
        max_budget = 15
        max_indicator = 3
        
        self.optimization.problem.max_budget = max_budget
        self.optimization.problem.max_indicator = max_indicator
        
        np.random.seed(2)
        random.seed(2)
        res = self.optimization.minimize()  
        
        self.assertTrue(res.G[0][1] <= 0.3)
        cost = res.F.T[1]
        self.assertTrue(max(cost) < max_budget)


   
if __name__ == '__main__':
    unittest.main()