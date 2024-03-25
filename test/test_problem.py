import unittest
import random
import numpy as np

from ams.prediction.markov import MarkovContinous

from ams.performance.performance import Performance

from ams.optimization.problem import MaintenanceSchedulingProblem, MultiIndicatorProblem, NetworkProblem
   
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
        actions = self.problem._decode_solution(np.array([0, 0] * 5))
        self.assertEqual(actions,{})
        
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
        
    def test_calc_area_under_curve_with_different_initial_IC(self):
        # Testing the area under the curve calculation
        random.seed(1)
        self.problem.initial_IC = 3
        performance = self.problem._evaluate_performance([self.action_binary])[0]
        area_under_curve = self.problem._calc_area_under_curve([performance])[0]
        expected_area = 65.4  # Expected area value based on the performance model
        self.assertAlmostEqual(area_under_curve, expected_area, delta=1e-5)
        self.problem.initial_IC = None # Reset value
    
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

    def test_calc_area_under_curve_with_different_initial_ICs(self):
        self.problem.initial_ICs = {'Bearing_Capacity':3,
                                    'Cracking':1,
                                    'Longitudinal_Evenness':2,
                                    'Skid_Resistance':2,
                                    'Transverse_Evenness':3,
                                    }
        random.seed(1)
        performance = self.problem._evaluate_performance([np.array([0, 0] * 5)])[0]
        area_under_curve = self.problem._calc_area_under_curve([performance])[0]

        self.assertAlmostEqual(area_under_curve['Bearing_Capacity'], 65, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Cracking'], 40., delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Longitudinal_Evenness'],  50.9, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Skid_Resistance'], 68.2, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Transverse_Evenness'], 76.6, delta=1e-5)

        random.seed(1)
        performance = self.problem._evaluate_performance([self.action_binary])[0]
        area_under_curve = self.problem._calc_area_under_curve([performance])[0]

        self.assertAlmostEqual(area_under_curve['Bearing_Capacity'], 65, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Cracking'], 33.9, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Longitudinal_Evenness'], 48.4, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Skid_Resistance'], 71.9, delta=1e-5)
        self.assertAlmostEqual(area_under_curve['Transverse_Evenness'], 51.8, delta=1e-5)
        
        self.problem.initial_ICs = {}
    
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
    
    def test_calc_max_indicators(self):
        random.seed(1)
        performance = self.problem._evaluate_performance([np.array([0, 0] * 5)])[0]
        indicators_diff = self.problem._calc_max_indicators([performance])
        
        self.assertAlmostEqual(indicators_diff[0][0], -3.7, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[1][0], -2, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[2][0], -2.7, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[3][0], -1.2, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[4][0], -2.2, delta=1e-5)

        random.seed(1)
        performance = self.problem._evaluate_performance([self.action_binary])[0]
        indicators_diff = self.problem._calc_max_indicators([performance])

        self.assertAlmostEqual(indicators_diff[0][0], -3.7, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[1][0], -2.3, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[2][0], -2.8, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[3][0], -1.6, delta=1e-5)
        self.assertAlmostEqual(indicators_diff[4][0], -2.8, delta=1e-5)
    
    def test_evaluate(self):
        out = {}
        random.seed(1)
        self.problem._evaluate([self.action_binary], out)

        self.assertAlmostEqual(out['F'][0][0], 52.5, delta=1e-5)
        self.assertAlmostEqual(out['F'][1][0], 16.0243, places=3)  

class TestNetworkProblem(unittest.TestCase):

    def setUp(self):        
        section_optimization = {
            "Asset_1": {
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
                        "15": "Corrective",
                        "30": "Corrective"
                    },
                    {
                        "10": "Preventive",
                        "30": "Corrective"
                    },
                    {
                        "20": "Preventive"
                    }
                ],
            },
            "Asset_2": {
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
                        "15": "Corrective",
                        "30": "Corrective"
                    },
                    {
                        "10": "Preventive",
                        "30": "Corrective"
                    },
                    {
                        "20": "Preventive"
                    }
                ],
            },
            "Asset_3": {
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
                        "15": "Corrective",
                        "30": "Corrective"
                    },
                    {
                        "10": "Preventive",
                        "30": "Corrective"
                    },
                    {
                        "20": "Preventive"
                    }
                ],
            },
            "Asset_4": {
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
                        "15": "Corrective",
                        "30": "Corrective"
                    },
                    {
                        "10": "Preventive",
                        "30": "Corrective"
                    },
                    {
                        "20": "Preventive"
                    }
                ],
            },
            "Asset_5": {
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
                        "15": "Corrective",
                        "30": "Corrective"
                    },
                    {
                        "10": "Preventive",
                        "30": "Corrective"
                    },
                    {
                        "20": "Preventive"
                    }
                ],
            },
        }
        
        self.problem = NetworkProblem(section_optimization)
        
    def test_calc_network_budget(self):
        population = [1, 1, 1, 1, 1]
        network_cost = self.problem._calc_network_budget(population)
        self.assertEqual(200, network_cost)
    
        population = [1, 0, 0, 1, 1]
        network_cost = self.problem._calc_network_budget(population)
        self.assertEqual(240, network_cost)
        
        population = [2, 2, 2, 2, 2]
        network_cost = self.problem._calc_network_budget(population)
        self.assertEqual(50, network_cost)
        
    def test_calc_network_budget_pop(self):
        population = [[1, 1, 1, 1, 1],
                      [1, 0, 0, 1, 1],
                      [2, 2, 2, 2, 2]
                      ]
        
        network_costs = self.problem._calc_network_budget_pop(population)
        np.testing.assert_array_equal(network_costs,[200,240,50])
        
    def test_calc_network_performance_indicator(self):
        population = [1, 1, 1, 1, 1]
        network_indicator = self.problem._calc_network_performance_indicator(population)
        self.assertEqual(50, network_indicator)
    
        population = [1, 0, 0, 1, 1]
        network_indicator = self.problem._calc_network_performance_indicator(population)
        self.assertEqual(38, network_indicator)
        
        population = [2, 2, 2, 2, 2]
        network_indicator = self.problem._calc_network_performance_indicator(population)
        self.assertEqual(100, network_indicator)
        
    def test_calc_network_performance_indicator_pop(self):
        population = [[1, 1, 1, 1, 1],
                      [1, 0, 0, 1, 1],
                      [2, 2, 2, 2, 2]
                      ]
        
        network_indicators = self.problem._calc_network_performance_indicator_pop(population)
        np.testing.assert_array_equal(network_indicators,[50,  38, 100])
    
    
    def test_evaluate(self):
        population = [1, 1, 1, 1, 1]
        out = {}
        
        self.problem._evaluate([population], out)
        
        self.assertEqual(out['F'][0][0], 50)
        self.assertEqual(out['F'][1][0], 200)
        
        #########################################
        
        population = [1, 0, 0, 1, 1]
        out = {}
        
        self.problem._evaluate([population], out)
        
        self.assertEqual(out['F'][0][0], 38)
        self.assertEqual(out['F'][1][0], 240)
        
        #########################################
        
        population =  [2, 2, 2, 2, 2]
        out = {}
        
        self.problem._evaluate([population], out)
        
        self.assertEqual(out['F'][0][0], 100)
        self.assertEqual(out['F'][1][0], 50)
 
   
if __name__ == '__main__':
    unittest.main()