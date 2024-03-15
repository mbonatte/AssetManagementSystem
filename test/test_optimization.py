import unittest
import random
import numpy as np

from ams.prediction.markov import MarkovContinous

from ams.performance.performance import Performance

from ams.optimization.problem import MaintenanceSchedulingProblem, MultiIndicatorProblem
from ams.optimization.multi_objective_optimization import Multi_objective_optimization
   
class TestMaintenanceSchedulingProblem(unittest.TestCase):

    def setUp(self):
        # Common setup for all tests in this class
        markov = MarkovContinous(worst_IC=5, best_IC=1)
        markov.theta = [0.508275, 0.410478, 0.281638, 0.247331, 0.      ]
        
        # Define actions for testing
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
        
        model = Performance(markov, actions)
        self.problem = MaintenanceSchedulingProblem(model, time_horizon=20)
        
        # Example binary action representation
        self.action_binary = np.array([0, 0] * 5)
        self.action_binary[2] = 5  # Year
        self.action_binary[3] = 1  # Action 1
        self.action_binary[4] = 7  # Year
        self.action_binary[5] = 1  # Action 1
        self.action_binary[6] = 10 # Year
        self.action_binary[7] = 2  # Action 2
        
    def test_decode_solution(self):
        # Testing the decoding of a solution
        decoded_actions = self.problem._decode_solution(self.action_binary)
        expected_actions = {'10': 'action_2', '5': 'action_1', '7': 'action_1'}
        self.assertEqual(decoded_actions, expected_actions)
                          
    def test_calc_budget(self):
        # Testing the budget calculation
        total_cost = self.problem._calc_budget([self.action_binary])[0]
        expected_cost = 16.0243  # Expected cost based on action costs and discount rates
        self.assertAlmostEqual(total_cost, expected_cost, places=3)
        
    def test_get_number_of_interventions(self):
        # Testing the count of maintenance interventions
        n_actions = self.problem._get_number_of_interventions(self.action_binary)
        expected_actions_count = 3  # Expected number of actions based on the test setup
        self.assertEqual(n_actions, expected_actions_count)
        
    def test_calc_area_under_curve(self):
        # Testing the area under the curve calculation
        random.seed(1)
        performance = self.problem._evaluate_performance([self.action_binary])[0]
        area_under_curve = self.problem._calc_area_under_curve([performance])[0]
        expected_area = 50.6  # Expected area value based on the performance model
        self.assertAlmostEqual(area_under_curve, expected_area, delta=1e-5)
    
    def test_calc_max_indicator(self):
        # Testing the maximum indicator calculation
        random.seed(1)
        performance = self.problem._evaluate_performance([self.action_binary])[0]
        max_indicator = self.problem._calc_max_indicator([performance])[0]
        expected_max_indicator = 3.8  # Expected maximum indicator value
        self.assertAlmostEqual(max_indicator, expected_max_indicator, delta=1e-5)
    
    def test_evaluate(self):
        # Testing the overall evaluation of the solution
        out = {}
        random.seed(1)
        self.problem._evaluate([self.action_binary], out)
        expected_performance = 50.6
        expected_cost = 16.0243
        self.assertAlmostEqual(out['F'][0][0], expected_performance, delta=1e-5)
        self.assertAlmostEqual(out['F'][1][0], expected_cost, places=3)
    
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

class TestMultiIndicatorProblem(unittest.TestCase):

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
                   "cost": 10
                   },
        ]

        # Create one performance model for each indicator
        performance_models = {}
        for key, theta in thetas.items():
            markov = MarkovContinous(worst_IC=5, best_IC=1)
            markov.theta = theta
            filtered_actions = MultiIndicatorProblem.extract_indicator(key, actions)
            performance_models[key] = Performance(markov, filtered_actions)

        self.problem = MultiIndicatorProblem(performance_models, time_horizon=20)

        # Example binary action representation
        self.action_binary = np.array([0, 0] * 5)
        self.action_binary[2] = 5  # Year
        self.action_binary[3] = 1  # Action 1
        self.action_binary[4] = 7  # Year
        self.action_binary[5] = 1  # Action 1
        self.action_binary[6] = 10 # Year
        self.action_binary[7] = 2  # Action 2

    def test_decode_solution(self):
        actions = self.problem._decode_solution(np.array([0, 0] * 5))

        self.assertEqual(actions,{})

        actions = self.problem._decode_solution(self.action_binary)

        self.assertEqual(actions,
                         {
                          '10': 'action_2',
                          '5': 'action_1',
                          '7': 'action_1'})

    def test_calc_budget(self):
        total_cost = self.problem._calc_budget([self.action_binary])[0]
        self.assertAlmostEqual(total_cost, 16.0243, places=3)

    def test_get_number_of_interventions(self):
        n_actions = self.problem._get_number_of_interventions(self.action_binary)
        self.assertEqual(n_actions, 3)

    def test_calc_area_under_curve(self):
        random.seed(1)
        performance = self.problem._evaluate_performance([np.array([0, 0] * 5)])[0]
        area_under_curve = self.problem._calc_area_under_curve([performance])[0]

        self.assertAlmostEqual(area_under_curve['Bearing_Capacity'], 23.7, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Cracking'], 40., delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Longitudinal_Evenness'], 36, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Skid_Resistance'], 55.5, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Transverse_Evenness'], 41.5, delta=1e-5)

        random.seed(1)
        performance = self.problem._evaluate_performance([self.action_binary])[0]
        area_under_curve = self.problem._calc_area_under_curve([performance])[0]

        self.assertAlmostEqual(area_under_curve['Bearing_Capacity'], 23.7, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Cracking'], 33.9, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Longitudinal_Evenness'], 32.3, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Skid_Resistance'], 52.5, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Transverse_Evenness'], 30, delta=1e-5)

    def test_calc_max_indicator(self):
        random.seed(1)
        performance = self.problem._evaluate_performance([np.array([0, 0] * 5)])[0]
        max_indicator = self.problem._calc_max_indicator([performance])[0]

        self.assertAlmostEqual(max_indicator['Bearing_Capacity'], 1.3, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Cracking'], 3, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Longitudinal_Evenness'], 2.3, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Skid_Resistance'], 3.8, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Transverse_Evenness'], 2.8, delta=1e-5)

        random.seed(1)
        performance = self.problem._evaluate_performance([self.action_binary])[0]
        max_indicator = self.problem._calc_max_indicator([performance])[0]

        self.assertAlmostEqual(max_indicator['Bearing_Capacity'], 1.3, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Cracking'], 2.7, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Longitudinal_Evenness'], 2.2, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Skid_Resistance'], 3.4, delta=1e-5)
        self.assertAlmostEqual(max_indicator['Transverse_Evenness'], 2.2, delta=1e-5)


    def test_evaluate(self):
        out = {}
        random.seed(1)
        self.problem._evaluate([self.action_binary], out)

        self.assertAlmostEqual(out['F'][0][0], 52.5, delta=1e-5)
        self.assertAlmostEqual(out['F'][1][0], 16.0243, places=3)  

   
if __name__ == '__main__':
    unittest.main()