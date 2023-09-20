import unittest
import random
import numpy as np

from prediction.markov import MarkovContinous

from optimization.problem import MaintenanceSchedulingProblem
from optimization.multi_objective_optimization import Multi_objective_optimization

# from numba import njit
# @njit
# def set_seed(value):
    # np.random.seed(value)

class TestMaintenanceSchedulingProblem(unittest.TestCase):

    def setUp(self):
        markov = MarkovContinous(worst_IC=5, best_IC=1)
        markov.theta = [0.508275, 0.410478, 0.281638, 0.247331, 0.      ]
        
        actions = [{"name": 'action_1',
                   "time_of_reduction": [
                                        0, 0, 0, 
                                        5, 5, 5, 
                                        5, 5, 5, 
                                        0, 0, 0, 
                                        0, 0, 0
                                    ],
                   "reduction_rate":    [
                                        0, 0, 0, 
                                        0.1, 0.1, 0.1,
                                        0.1, 0.1, 0.1,
                                        0, 0, 0, 
                                        0, 0, 0
                                    ],
                   "cost": 3.70
                   },
                   {"name": 'action_2',
                   "time_of_reduction": [
                                        0, 0, 0, 
                                        5, 5, 5, 
                                        5, 5, 5, 
                                        0, 0, 0, 
                                        0, 0, 0
                                    ],
                   "reduction_rate": [
                                    0, 0, 0, 
                                    0.1, 0.1, 0.1,
                                    0.1, 0.1, 0.1,
                                    0, 0, 0, 
                                    0, 0, 0
                                    ],
                   "cost": 3.70
                   },
        ]
        
        time_horizon = 20
        
        self.problem = MaintenanceSchedulingProblem(markov, actions, time_horizon)
        self.problem.performance_model._number_of_process = 1
        
        self.action_binary = np.array([0, 0] * 5)
        
        self.action_binary[2] = 5 #year
        self.action_binary[3] = 1 #action_1
        
        self.action_binary[4] = 7 #year
        self.action_binary[5] = 1 #action_1
        
        self.action_binary[6] = 10 #year
        self.action_binary[7] = 2 #action_2
        
    def test_decode_solution(self):
        actions = self.problem._decode_solution(self.action_binary)
        
        self.assertEqual(actions,
                         {'0': 'DoNothing',
                          '10': 'action_2',
                          '5': 'action_1',
                          '7': 'action_1'})
                          
    def test_calc_budget(self):
        total_cost = self.problem._calc_budget([self.action_binary])[0]
        self.assertAlmostEqual(total_cost, 10.321, places=3)
        
    def test_get_number_of_interventions(self):
        n_actions = self.problem._get_number_of_interventions(self.action_binary)
        self.assertEqual(n_actions, 3)
        
    def test_calc_performance(self):
        random.seed(1)
        # set_seed(1)
        area_under_curve = self.problem._calc_performance([self.action_binary])[0]
        self.assertAlmostEqual(area_under_curve, 65.10000, delta=1e-5)
    
    def test_evaluate(self):
        out = {}
        random.seed(1)
        # set_seed(1)
        self.problem._evaluate([self.action_binary], out)
        self.assertAlmostEqual(out['F'][0][0], 65.10000, delta=1e-5)
        self.assertAlmostEqual(out['F'][1][0], 10.321, places=3)
    

class TestMulti_objective_optimization(unittest.TestCase):
    def setUp(self):
        markov = MarkovContinous(worst_IC=5, best_IC=1)
        markov.theta = [0.508275, 0.410478, 0.281638, 0.247331, 0.      ]
        
        actions = [{"name": 'action_1',
                   "time_of_reduction": [
                                        0, 0, 0, 
                                        5, 5, 5, 
                                        5, 5, 5, 
                                        0, 0, 0, 
                                        0, 0, 0
                                    ],
                   "reduction_rate":    [
                                        0, 0, 0, 
                                        0.1, 0.1, 0.1,
                                        0.1, 0.1, 0.1,
                                        0, 0, 0, 
                                        0, 0, 0
                                    ],
                   "cost": 3.70
                   },
                   {"name": 'action_2',
                   "time_of_reduction": [
                                        0, 0, 0, 
                                        5, 5, 5, 
                                        5, 5, 5, 
                                        0, 0, 0, 
                                        0, 0, 0
                                    ],
                   "reduction_rate": [
                                    0, 0, 0, 
                                    0.1, 0.1, 0.1,
                                    0.1, 0.1, 0.1,
                                    0, 0, 0, 
                                    0, 0, 0
                                    ],
                   "cost": 3.70
                   },
        ]
        
        time_horizon = 20
        
        problem = MaintenanceSchedulingProblem(markov, actions, time_horizon)
        problem.performance_model._number_of_process = 1
        
        self.optimization = Multi_objective_optimization()
        self.optimization.verbose = False
        self.optimization.set_problem(problem)
        
    def test_minimize(self):
        # set_seed(1)
        np.random.seed(1)
        res = self.optimization.minimize()
        
        sort = np.argsort(res.F.T)[1]
        
        performance = res.F.T[0][sort]
        cost = res.F.T[1][sort]
        best_action = self.optimization.problem._decode_solution(res.X[sort][-1])
        
        self.assertAlmostEqual(performance[0], 77.8999, places=3)
        self.assertAlmostEqual(performance[-1], 50.9, places=3)
        
        self.assertAlmostEqual(cost[0], 0, places=3)
        self.assertAlmostEqual(cost[-1], 13.816831, places=5)
        
        action =  {'11': 'DoNothing', '13': 'action_1', '2': 'action_2', '4': 'action_1', '9': 'action_1'}
        self.assertEqual(action, best_action)

if __name__ == '__main__':
    unittest.main()
