import unittest
import warnings
import numpy as np

from ams.optimization.problem import NetworkProblem
from ams.optimization.multi_objective_optimization import Multi_objective_optimization

class TestNetworkOptimization(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning, module='pymoo')
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