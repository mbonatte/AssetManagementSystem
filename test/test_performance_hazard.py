import unittest
import random
import numpy as np

from ams.prediction.markov import MarkovContinous
from ams.performance.hazard_effects import HazardEffect
from ams.performance.performance import Performance

    
class TestPerformance(unittest.TestCase):

    def setUp(self):
        markov = MarkovContinous(worst_IC=5, best_IC=1)
        markov.theta = [0.508275, 0.410478, 0.281638, 0.247331, 0.      ]
        
        action = [
            {
                "name": 'light',
                "time_of_reduction": {
                    1: [5, 5, 5],
                    2: [5, 5, 5],
                    3: [5, 5, 5],
                    4: [5, 5, 5],
                    5: [5, 5, 5],
                },
                "reduction_rate": {
                    1: [0.1, 0.1, 0.1],
                    2: [0.1, 0.1, 0.1],
                    3: [0.1, 0.1, 0.1],
                    4: [0.1, 0.1, 0.1],
                    5: [0.1, 0.1, 0.1],
                }
            },
            {
                "name": 'improvement',
                "improvement": {
                    2: [5, 5, 5],
                    3: [5, 5, 5],
                    4: [5, 5, 5],
                    5: [5, 5, 5]
                },
            }
        ]

        hazard_data = [
            {
                "name": "Light Damage",
                "probability": 0.161281,
                "increase_rate":   {
                    "1": [5, 5, 5],
                    "2": [5, 5, 5],
                    "3": [5, 5, 5],
                    "4": [5, 5, 5],
                    "5": [5, 5, 5],
                },
                "time_of_increase":   {
                    "1": [5, 5, 5],
                    "2": [5, 5, 5],
                    "3": [5, 5, 5],
                    "4": [5, 5, 5],
                    "5": [5, 5, 5],
                }
            },
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

        
        self.performance = Performance(markov,
                                       action,
                                       hazard_data)
                                       
    
    def test_get_hazard(self):
        np.random.seed(1)
        hazards = self.performance._sample_hazards(50)
        self.assertEqual(hazards[14], 'Minor Damage')

    def test_get_next_IC(self):
        np.random.seed(42)
        IC = self.performance._get_next_IC(2, {}, 14, {})
        self.assertEqual(IC, 2)

        np.random.seed(1)
        IC = self.performance._get_next_IC(2, {}, 14, {14: 'Minor Damage'})
        self.assertEqual(IC, 3)

        np.random.seed(1)
        self.performance.hazard_effects = {'Minor Damage': HazardEffect('Minor Damage')}
        IC = self.performance._get_next_IC(2, {}, 14, {14: 'Minor Damage'})
        self.assertEqual(IC, 2)
       
    def test_get_degradated_IC(self):
        self.assertEqual(self.performance.get_degradated_IC(10, 0),
                         5)
        self.assertEqual(self.performance.get_degradated_IC(3, 0),
                         3)
        self.assertEqual(self.performance.get_degradated_IC(3, 1),
                         4)
        self.assertEqual(self.performance.get_degradated_IC(1, 2),
                         3)
        self.assertEqual(self.performance.get_degradated_IC(1, 10),
                         5)
    
    def test_get_IC_over_time(self):
        time_hoziron = 5
        initial_IC = 1
        maintenance_scenario = {}
        
        random.seed(1)
        np.random.seed(1)
        IC = self.performance.get_IC_over_time(time_hoziron,
                                               initial_IC,
                                               actions_schedule=maintenance_scenario,
                                               number_of_samples=10)
        
        self.assertEqual(IC[-1], 4.3)

        time_hoziron = 20
        initial_IC = 1
        maintenance_scenario = {}
        
        random.seed(1)
        IC = self.performance.get_IC_over_time(time_hoziron,
                                               initial_IC,
                                               actions_schedule=maintenance_scenario,
                                               number_of_samples=10)
        
        self.assertEqual(IC[-1], 5)

    def test_get_IC_over_time_with_hazards_schedule(self):
        time_hoziron = 2
        initial_IC = 1
        hazards_schedule = {1: 'Collapse'}
        
        random.seed(1)
        np.random.seed(1)
        IC = self.performance.get_IC_over_time(time_hoziron,
                                               initial_IC,
                                               actions_schedule={},
                                               hazards_schedule=hazards_schedule,
                                               number_of_samples=1)
        
        self.assertEqual(IC[-1], 5)
    
        ###############################
        time_hoziron = 5
        initial_IC = 1
        hazards_schedule = {1: 'Light Damage'}
        
        random.seed(1)
        np.random.seed(1)
        IC = self.performance.get_IC_over_time(time_hoziron,
                                               initial_IC,
                                               actions_schedule={},
                                               hazards_schedule=hazards_schedule,
                                               number_of_samples=1)
        
        self.assertEqual(IC[-1], 5)

    
    def test_get_IC_over_time_actions_hazards_schedule(self):
        time_hoziron = 10
        initial_IC = 1
        maintenance_scenario = {
            '3': 'light',
            '5': 'improvement',
            '8': 'light',
            '10': 'improvement',
            # '15': 'improvement',
            # '20': 'improvement',
            }
        hazards_schedule = {
            2: 'Light Damage', 
            7: 'Light Damage', 
            # 2: 'Moderate Damage', 
            # 3: 'Severe Damage', 
            # 4: 'Collapse'
            }
        
        random.seed(1)
        np.random.seed(1)
        IC = self.performance.get_IC_over_time(time_hoziron,
                                               initial_IC,
                                               actions_schedule=maintenance_scenario,
                                               hazards_schedule=hazards_schedule,
                                               number_of_samples=1)
        
        self.assertEqual(IC[2], 4)
        self.assertEqual(IC[-2], 3)

    
if __name__ == '__main__':
    unittest.main()
