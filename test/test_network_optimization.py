import unittest
import random
import numpy as np

from optimization.problem import NetworkProblem, NetworkTrafficProblem
from optimization.multi_objective_optimization import Multi_objective_optimization

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

class TestNetworkOptimization_Real(unittest.TestCase):
    def setUp(self):
        import json
        from pathlib import Path
        
        MAIN_FOLDER = Path(__file__).parent.parent.resolve()
        
        path = MAIN_FOLDER / 'database/optimization_output.json'
        with open(path, "r") as file:
            section_optimization = json.load(file)
        
        problem = NetworkProblem(section_optimization)
        
        self.optimization = Multi_objective_optimization()
        self.optimization.verbose = False
        self.optimization.set_problem(problem)
        
        self.optimization._set_algorithm({"name": "NSGA2", "pop_size": 20, "eliminate_duplicates": True})
        self.optimization._set_termination({'name':'n_gen', 'n_max_gen':20})
        
    def test_minimize(self):
        np.random.seed(1)
        
        res = self.optimization.minimize()
        
        sort = np.argsort(res.F.T)[1]
        
        performance = res.F.T[0][sort]
        cost = res.F.T[1][sort]
        
        self.assertEqual(performance[0], 93.90625)
        self.assertEqual(performance[-1], 60.3375)
        
        self.assertEqual(cost[0], 146.94)
        self.assertEqual(cost[-1], 407.52)

class TestNetworkTrafficProblem(unittest.TestCase):

    def setUp(self):
        actions = [
            {
            "name": "Minor",
            "cost": 10
            },
            {
            "name": "Major",
            "cost": 30
            },
        ]
        
        section_optimization = {
            "Road_1": {
                "Performance": [
                    20,
                    50,
                    100
                ],
                "Cost": [
                    60,
                    40,
                    10
                ],
                "Actions_schedule": [
                    {
                        "15": "Major",
                        "30": "Major"
                    },
                    {
                        "10": "Minor",
                        "30": "Major"
                    },
                    {
                        "20": "Minor"
                    }
                ],
            },
            "Road_2_1": {
                "Performance": [
                    20,
                    50,
                    100
                ],
                "Cost": [
                    60,
                    40,
                    10
                ],
                "Actions_schedule": [
                    {
                        "15": "Major",
                        "30": "Major"
                    },
                    {
                        "10": "Minor",
                        "30": "Major"
                    },
                    {
                        "20": "Minor"
                    }
                ],
            },
            "Road_2_2": {
                "Performance": [
                    20,
                    50,
                    100
                ],
                "Cost": [
                    60,
                    40,
                    10
                ],
                "Actions_schedule": [
                    {
                        "15": "Major",
                        "30": "Major"
                    },
                    {
                        "10": "Minor",
                        "30": "Major"
                    },
                    {
                        "20": "Minor"
                    }
                ],
            },
            "Road_3": {
                "Performance": [
                    20,
                    50,
                    100
                ],
                "Cost": [
                    60,
                    40,
                    10
                ],
                "Actions_schedule": [
                    {
                        "15": "Major",
                        "30": "Major"
                    },
                    {
                        "10": "Minor",
                        "30": "Major"
                    },
                    {
                        "20": "Minor"
                    }
                ],
            },
            "Road_4": {
                "Performance": [
                    20,
                    50,
                    100
                ],
                "Cost": [
                    60,
                    40,
                    10
                ],
                "Actions_schedule": [
                    {
                        "15": "Major",
                        "30": "Major"
                    },
                    {
                        "10": "Minor",
                        "30": "Major"
                    },
                    {
                        "20": "Minor"
                    }
                ],
            },
        }
        
        TMS_output = {
            "Road_1": {
                "Minor_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Minor_2": {
                    "Fuel": 1000000,
                    "Cost": 0.8
                    },
                "Major_10": {
                    "Fuel": 50,
                    "Cost": 1
                    },
                "Major_2": {
                    "Fuel": 1000000,
                    "Cost": 0.8
                    },
                },
            "Road_2_1": {
                "Minor_10": {
                    "Fuel": 10,
                    "Cost": 1
                    },
                "Minor_2": {
                    "Fuel": 20,
                    "Cost": 0.8
                    },
                "Major_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Major_2": {
                    "Fuel": 40,
                    "Cost": 0.8
                    },
                },
            "Road_2_2": {
                "Minor_10": {
                    "Fuel": 10,
                    "Cost": 1
                    },
                "Minor_2": {
                    "Fuel": 20,
                    "Cost": 0.8
                    },
                "Major_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Major_2": {
                    "Fuel": 40,
                    "Cost": 0.8
                    },
                },
            "Road_3": {
                "Minor_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Minor_2": {
                    "Fuel": 1000000,
                    "Cost": 0.8
                    },
                "Major_10": {
                    "Fuel": 50,
                    "Cost": 1
                    },
                "Major_2": {
                    "Fuel": 1000000,
                    "Cost": 0.8
                    },
                },
            "Road_4": {
                "Minor_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Minor_2": {
                    "Fuel": 1000000,
                    "Cost": 0.8
                    },
                "Major_10": {
                    "Fuel": 50,
                    "Cost": 1
                    },
                "Major_2": {
                    "Fuel": 1000000,
                    "Cost": 0.8
                    },
                },
        }
        
        self.problem = NetworkTrafficProblem(section_optimization, TMS_output, actions)
        
    
    def test_calc_network_budget(self):
        population = [1, 1, 1, 1, 1]
        network_cost = self.problem._calc_network_budget(population)
        self.assertEqual(200, network_cost)
    
        population = [1, 4, 4, 1, 1]
        network_cost = self.problem._calc_network_budget(population)
        self.assertEqual(184, network_cost)
        
        population = [4, 4, 4, 4, 4]
        fuel = self.problem._calc_network_budget(population)
        self.assertEqual(160, fuel)
    
    def test_calc_fuel(self):
        population = [1, 1, 1, 1, 1]
        fuel = self.problem._calc_fuel(population)
        self.assertEqual(270, fuel)
        
        population = [1, 4, 4, 1, 1]
        fuel = self.problem._calc_fuel(population)
        self.assertEqual(330, fuel)
        
        population = [4, 4, 4, 4, 4]
        fuel = self.problem._calc_fuel(population)
        self.assertEqual(6000120, fuel)
    
    
    def test_evaluate(self):
        population = [1, 1, 1, 1, 1]
        out = {}
        
        self.problem._evaluate([population], out)
        
        self.assertEqual(out['F'][1][0], 200)
        self.assertEqual(out['F'][2][0], 270)
        
        #########################################
        
        population = [1, 4, 4, 1, 1]
        out = {}
        
        self.problem._evaluate([population], out)
        
        self.assertEqual(out['F'][1][0], 184)
        self.assertEqual(out['F'][2][0], 330)
        
        #########################################
        
        population = [4, 4, 4, 4, 4]
        out = {}
        
        self.problem._evaluate([population], out)
        
        self.assertEqual(out['F'][1][0], 160)
        self.assertEqual(out['F'][2][0], 6000120)
    
class TestNetworkTrafficOptimization(unittest.TestCase):
    def setUp(self):
        actions = [
            {
            "name": "Minor",
            "cost": 10
            },
            {
            "name": "Major",
            "cost": 30
            },
        ]
        
        section_optimization = {
            "Road_1": {
                "Performance": [
                    20,
                    50,
                    100
                ],
                "Cost": [
                    60,
                    40,
                    10
                ],
                "Actions_schedule": [
                    {
                        "15": "Major",
                        "30": "Major"
                    },
                    {
                        "10": "Minor",
                        "30": "Major"
                    },
                    {
                        "20": "Minor"
                    }
                ],
            },
            "Road_2_1": {
                "Performance": [
                    20,
                    50,
                    100
                ],
                "Cost": [
                    60,
                    40,
                    10
                ],
                "Actions_schedule": [
                    {
                        "15": "Major",
                        "30": "Major"
                    },
                    {
                        "10": "Minor",
                        "30": "Major"
                    },
                    {
                        "20": "Minor"
                    }
                ],
            },
            "Road_2_2": {
                "Performance": [
                    20,
                    50,
                    100
                ],
                "Cost": [
                    60,
                    40,
                    10
                ],
                "Actions_schedule": [
                    {
                        "15": "Major",
                        "30": "Major"
                    },
                    {
                        "10": "Minor",
                        "30": "Major"
                    },
                    {
                        "20": "Minor"
                    }
                ],
            },
            "Road_3": {
                "Performance": [
                    20,
                    50,
                    100
                ],
                "Cost": [
                    60,
                    40,
                    10
                ],
                "Actions_schedule": [
                    {
                        "15": "Major",
                        "30": "Major"
                    },
                    {
                        "10": "Minor",
                        "30": "Major"
                    },
                    {
                        "20": "Minor"
                    }
                ],
            },
            "Road_4": {
                "Performance": [
                    20,
                    50,
                    100
                ],
                "Cost": [
                    60,
                    40,
                    10
                ],
                "Actions_schedule": [
                    {
                        "15": "Major",
                        "30": "Major"
                    },
                    {
                        "10": "Minor",
                        "30": "Major"
                    },
                    {
                        "20": "Minor"
                    }
                ],
            },
        }
        
        TMS_output = {
            "Road_1": {
                "Minor_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Minor_2": {
                    "Fuel": 10000000,
                    "Cost": 0.8
                    },
                "Major_10": {
                    "Fuel": 50,
                    "Cost": 1
                    },
                "Major_2": {
                    "Fuel": 10000000,
                    "Cost": 0.8
                    },
                },
            "Road_2_1": {
                "Minor_10": {
                    "Fuel": 10,
                    "Cost": 1
                    },
                "Minor_2": {
                    "Fuel": 20,
                    "Cost": 0.3
                    },
                "Major_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Major_2": {
                    "Fuel": 40,
                    "Cost": 0.3
                    },
                },
            "Road_2_2": {
                "Minor_10": {
                    "Fuel": 10,
                    "Cost": 1
                    },
                "Minor_2": {
                    "Fuel": 20,
                    "Cost": 0.3
                    },
                "Major_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Major_2": {
                    "Fuel": 40,
                    "Cost": 0.3
                    },
                },
            "Road_3": {
                "Minor_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Minor_2": {
                    "Fuel": 10000000,
                    "Cost": 0.8
                    },
                "Major_10": {
                    "Fuel": 50,
                    "Cost": 1
                    },
                "Major_2": {
                    "Fuel": 10000000,
                    "Cost": 0.8
                    },
                },
            "Road_4": {
                "Minor_10": {
                    "Fuel": 20,
                    "Cost": 1
                    },
                "Minor_2": {
                    "Fuel": 10000000,
                    "Cost": 0.8
                    },
                "Major_10": {
                    "Fuel": 50,
                    "Cost": 1
                    },
                "Major_2": {
                    "Fuel": 10000000,
                    "Cost": 0.8
                    },
                },
        }
        
        problem = NetworkTrafficProblem(section_optimization, TMS_output, actions)
        
        self.optimization = Multi_objective_optimization()
        self.optimization.verbose = False
        self.optimization.set_problem(problem)
        
        self.optimization._set_algorithm({"name": "NSGA2", "pop_size": 30, "eliminate_duplicates": True})
        self.optimization._set_termination({'name':'n_gen', 'n_max_gen':200})
        
    def test_minimize(self):
        np.random.seed(1)
        
        res = self.optimization.minimize()
        
        sort = np.argsort(res.F.T)[1]
        
        filter_fuel = res.F.T[2] < 10000000
        
        performance = res.F.T[0][filter_fuel]
        cost = res.F.T[1][filter_fuel]
        fuel = res.F.T[2][filter_fuel]
        
        sort_fuel = np.argsort(fuel)
        
        # print(performance)
        # print(cost)
        # print(fuel)
        
        # print(res.X[filter_fuel][sort_fuel])
        
        # from mpl_toolkits import mplot3d
        # from mpl_toolkits.mplot3d import Axes3D
        # from matplotlib import cm
        # import matplotlib.pyplot as plt
        
        # fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        
        # ax[0].set_xlabel('performance')
        # ax[0].set_ylabel('cost')
        # for i in range(len(performance)):
            # ax[0].plot(performance[i], cost[i], 'o')
        
        # ax[1].set_xlabel('performance')
        # ax[1].set_ylabel('fuel')
        # for i in range(len(cost)):
            # ax[1].plot(performance[i], fuel[i], 'o')
        
        # ax[2].set_xlabel('cost')
        # ax[2].set_ylabel('fuel')
        # for i in range(len(fuel)):
            # ax[2].plot(cost[i], fuel[i], 'o')
        
        # # Creating figure
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # for p, c, f in zip(performance, cost, fuel):
            # ax.scatter(p, c, f, marker='o')
        # ax.set_xlabel('Performance')
        # ax.set_ylabel('Cost')
        # ax.set_zlabel('Fuel')
        
        # plt.show()
        
        # self.assertEqual(performance[0], 100)
        # self.assertEqual(performance[-1], 20)
        
        # self.assertEqual(cost[0], 20)
        # self.assertEqual(cost[-1], 100)    


if __name__ == '__main__':
    unittest.main()