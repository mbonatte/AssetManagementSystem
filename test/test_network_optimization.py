import unittest
import random
import numpy as np

from ams.optimization.problem import NetworkProblem
from ams.optimization.multi_objective_optimization import Multi_objective_optimization

class TestNetworkProblem(unittest.TestCase):

    def setUp(self):
        section_optimization = {
            "section_1": {
                "Performance": [
                    20,
                    40,
                    60,
                    80,
                    100
                ],
                "Cost": [
                    47,
                    35,
                    29,
                    21,
                    13
                ]
            },
            "section_2": {
                "Performance": [
                    20,
                    40,
                    60,
                    80,
                    100
                ],
                "Cost": [
                    50,
                    40,
                    30,
                    20,
                    10
                ]
            }
        }
        
        self.problem = NetworkProblem(section_optimization)
        
        
    def test_calc_network_performance_indicator(self):
        population = [2, 4]
        network_performance = self.problem._calc_network_performance_indicator(population)
        self.assertEqual(80, network_performance)
    
    def test_calc_network_budget(self):
        population = [2, 4]
        network_cost = self.problem._calc_network_budget(population)
        self.assertEqual(39, network_cost)
    
    def test_evaluate(self):
        population = [2, 4]
        out = {}
        
        self.problem._evaluate([population], out)

        self.assertEqual(out['F'][0][0], 80)
        self.assertEqual(out['F'][1][0], 39)

class TestNetworkOptimization(unittest.TestCase):
    def setUp(self):
        section_optimization = {
            "section_1": {
                "Performance": [
                    20,
                    40,
                    60,
                    80,
                    100
                ],
                "Cost": [
                    50,
                    40,
                    30,
                    20,
                    10
                ]
            },
            "section_2": {
                "Performance": [
                    20,
                    40,
                    60,
                    80,
                    100
                ],
                "Cost": [
                    50,
                    40,
                    30,
                    20,
                    10
                ]
            }
        }
        
        problem = NetworkProblem(section_optimization)
        
        self.optimization = Multi_objective_optimization()
        self.optimization.verbose = False
        self.optimization.set_problem(problem)
        
        self.optimization._set_algorithm({"name": "NSGA2", "pop_size": 20, "eliminate_duplicates": True})
        self.optimization._set_termination({'name':'n_gen', 'n_max_gen':3})
        
    def test_minimize(self):
        np.random.seed(1)
        
        res = self.optimization.minimize()
        
        sort = np.argsort(res.F.T)[1]
        
        performance = res.F.T[0][sort]
        cost = res.F.T[1][sort]
        
        self.assertEqual(performance[0], 100)
        self.assertEqual(performance[-1], 20)
        
        self.assertEqual(cost[0], 20)
        self.assertEqual(cost[-1], 100)

if __name__ == '__main__':
    unittest.main()