import unittest
import random
import numpy as np

from ams.prediction.markov import MarkovContinous

from ams.performance.maintenance import ActionEffect
from ams.performance.performance import Performance, Sample

class TestActionEffect(unittest.TestCase):

    def setUp(self):
        number_of_states = 5
        name = 'test'
        self.action = ActionEffect(name, number_of_states)
        
        
    def test_get_effect(self):
        current_state = 3
        num_samples = 10
        
        # ################################################################
        
        a, b, c = 0, 4, 4
        effect = {i:[a, b, c] for i in range(self.action.number_of_states)}
        self.action.set_improvement(effect)
        
        np.random.seed(1)
        counts = [self.action.get_improvement(current_state) 
                  for i in range(num_samples)]
        
        self.assertAlmostEqual(np.mean(counts), 2.0501, places=4)
        
        # ################################################################
        
        a, b, c = .7, .8, .9
        effect = {i:[a, b, c] for i in range(self.action.number_of_states)}
        self.action.set_reduction_rate(effect)
        
        np.random.seed(1)
        counts = [self.action.get_reduction_rate(current_state) 
                  for i in range(num_samples)]
        
        self.assertAlmostEqual(np.mean(counts), 0.7730, places=4)
        
        # ################################################################
        
        a, b, c = 4, 8, 9
        effect = {i:[a, b, c] for i in range(self.action.number_of_states)}
        self.action.set_time_of_delay(effect)
        
        np.random.seed(1)
        counts = [self.action.get_time_of_delay(current_state) 
                  for i in range(num_samples)]
        
        self.assertAlmostEqual(np.mean(counts), 6.2921, places=4)
        
        # ################################################################
        
        a, b, c = 2, 2, 9
        effect = {i:[a, b, c] for i in range(self.action.number_of_states)}
        self.action.set_time_of_reduction(effect)
        
        np.random.seed(1)
        counts = [self.action.get_time_of_reduction(current_state) 
                  for i in range(num_samples)]
        
        self.assertAlmostEqual(np.mean(counts), 3.2804, places=4)
        
    def test_set_effect(self):
        effect = {i:[0, 4, 4]
                  for i in range(self.action.number_of_states)}
        self.action.set_improvement(effect)
        self.assertEqual(self.action.improvement, effect)
        
        effect = {i:[0.8, 0.9, 1.1]
                  for i in range(self.action.number_of_states)}
        self.action.set_reduction_rate(effect)
        self.assertEqual(self.action.reduction_rate, effect)
        
        effect = {i:[1, 1, 4]
                  for i in range(self.action.number_of_states)}
        self.action.set_time_of_delay(effect)
        self.assertEqual(self.action.time_of_delay, effect)
        
        effect = {i:[7, 9, 10 ]
                  for i in range(self.action.number_of_states)}
        self.action.set_time_of_reduction(effect)
        self.assertEqual(self.action.time_of_reduction, effect)
        
    def test_set_action_effects(self):
        name = 'test'
        number_of_states = 5
        
        time_of_reduction = {
                                2: [2, 2, 2],
                                3: [1, 1, 1]
                            }
        
        reduction_rate = {
                            2: [0.2, 0.2, 0.2],
                            3: [0.2, 0.2, 0.2]
                            }
        
        action = [{"name": name,
                  "time_of_reduction": time_of_reduction,
                  "reduction_rate": reduction_rate
        }]
        action_effects = ActionEffect.set_action_effects(number_of_states,
                                                         action)
        
        self.assertEqual(action_effects[name].time_of_reduction,
                                             time_of_reduction)
        
        self.assertEqual(action_effects[name].reduction_rate,
                                             reduction_rate)
        
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
        
        self.performance = Performance(markov,
                                       action)
        
        self.performance._number_of_process = 1
                                       
    def test_get_improved_IC(self):
        self.assertEqual(self.performance.get_improved_IC(10, 0),
                         10)
        self.assertEqual(self.performance.get_improved_IC(10, 2),
                         8)
        self.assertEqual(self.performance.get_improved_IC(3, 2),
                         1)
        self.assertEqual(self.performance.get_improved_IC(3, 5),
                         1)
    
    def test_get_IC_over_time(self):
        time_hoziron = 20
        initial_IC = 1
        maintenance_scenario = {}
        
        random.seed(1)
        # set_seed(41)
        IC = self.performance.get_IC_over_time(time_hoziron,
                                               initial_IC,
                                               actions_schedule=maintenance_scenario,
                                               number_of_samples=10)
        
        self.assertEqual(IC[-1], 4.8)
        
        maintenance_scenario = {'1': 'test',
                                '4': 'test',
                                '10': 'test',
                                '15': 'test',}
        
        random.seed(1)
        # # set_seed(40)
        IC = self.performance.get_IC_over_time(time_hoziron,
                                               initial_IC,
                                               actions_schedule=maintenance_scenario,
                                               number_of_samples=10)
        self.assertEqual(IC[-1], 3.6)
    
    
if __name__ == '__main__':
    unittest.main()
