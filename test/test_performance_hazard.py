import unittest
import random
import numpy as np

from ams.prediction.markov import MarkovContinous

from ams.performance.maintenance import ActionEffect
from ams.performance.performance import Performance, Sample

    
class TestPerformance(unittest.TestCase):

    def setUp(self):
        markov = MarkovContinous(worst_IC=5, best_IC=1)
        markov.theta = [0.508275, 0.410478, 0.281638, 0.247331, 0.      ]
        
        name = 'test'
        
        time_of_reduction = {
                                2: [5, 5, 5],
                                3: [5, 5, 5]
                            }
        
        reduction_rate = {
                            2: [0.1, 0.1, 0.1],
                            3: [0.1, 0.1, 0.1]
                         }
        action = [{"name": name,
                  "time_of_reduction": time_of_reduction,
                  "reduction_rate": reduction_rate
        }]

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

        
        self.performance = Performance(markov,
                                       action,
                                       hazard_data)
                                       
    
    def test_get_hazard(self):
        np.random.seed(1)
        hazards = self.performance.get_hazards(50)
        self.assertEqual(hazards[13], 'Minor Damage')
       
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
        time_hoziron = 20
        initial_IC = 1
        maintenance_scenario = {}
        
        random.seed(1)
        IC = self.performance.get_IC_over_time(time_hoziron,
                                               initial_IC,
                                               actions_schedule=maintenance_scenario,
                                               number_of_samples=10)
        
        self.assertEqual(IC[-1], 5)
    
    
if __name__ == '__main__':
    unittest.main()
