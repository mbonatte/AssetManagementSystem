import unittest
import random
import numpy as np

from ams.prediction.markov import MarkovContinous

from ams.performance.maintenance import ActionEffect
from ams.performance.performance import Performance, Sample

class TestActionEffect(unittest.TestCase):

    def setUp(self):
        name = 'test'
        self.number_of_states = 5
        self.action = ActionEffect(name)
        
        
    def test_get_effect(self):
        current_state = 3
        num_samples = 10
        
        # ################################################################
        
        a, b, c = 0, 4, 4
        effect = {i:[a, b, c] for i in range(self.number_of_states)}
        self.action.set_improvement(effect)
        
        np.random.seed(1)
        counts = [self.action.get_improvement(current_state) 
                  for i in range(num_samples)]
        
        self.assertAlmostEqual(np.mean(counts), 2.0501, places=4)
        
        # ################################################################
        
        a, b, c = .7, .8, .9
        effect = {i:[a, b, c] for i in range(self.number_of_states)}
        self.action.set_reduction_rate(effect)
        
        np.random.seed(1)
        counts = [self.action.get_reduction_rate(current_state) 
                  for i in range(num_samples)]
        
        self.assertAlmostEqual(np.mean(counts), 0.7730, places=4)
        
        # ################################################################
        
        a, b, c = 4, 8, 9
        effect = {i:[a, b, c] for i in range(self.number_of_states)}
        self.action.set_time_of_delay(effect)
        
        np.random.seed(1)
        counts = [self.action.get_time_of_delay(current_state) 
                  for i in range(num_samples)]
        
        self.assertAlmostEqual(np.mean(counts), 6.2921, places=4)
        
        # ################################################################
        
        a, b, c = 2, 2, 9
        effect = {i:[a, b, c] for i in range(self.number_of_states)}
        self.action.set_time_of_reduction(effect)
        
        np.random.seed(1)
        counts = [self.action.get_time_of_reduction(current_state) 
                  for i in range(num_samples)]
        
        self.assertAlmostEqual(np.mean(counts), 3.2804, places=4)
        
    def test_set_effect(self):
        effect = {i:[0, 4, 4]
                  for i in range(self.number_of_states)}
        self.action.set_improvement(effect)
        self.assertEqual(self.action.improvement, effect)
        
        effect = {i:[0.8, 0.9, 1.1]
                  for i in range(self.number_of_states)}
        self.action.set_reduction_rate(effect)
        self.assertEqual(self.action.reduction_rate, effect)
        
        effect = {i:[1, 1, 4]
                  for i in range(self.number_of_states)}
        self.action.set_time_of_delay(effect)
        self.assertEqual(self.action.time_of_delay, effect)
        
        effect = {i:[7, 9, 10 ]
                  for i in range(self.number_of_states)}
        self.action.set_time_of_reduction(effect)
        self.assertEqual(self.action.time_of_reduction, effect)
        
    def test_set_action_effects(self):
        name = 'test'
        
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
        action_effects = ActionEffect.set_action_effects(action)
        
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
                                       
    
    def test_get_action(self):
        # Test the get_action method with different time inputs
        # Example:
        self.performance._set_actions_schedule({'10': 'action1', '20': 'action2'})
        self.assertEqual(self.performance.get_action(10), 'action1')
        self.assertIsNone(self.performance.get_action(5))
    
    def test_get_reduction_factor(self):
        sample = Sample()
        
        sample.timeOfReduction = 5
        sample.rateOfReduction = 0.8
        
        interventions = {10: sample}
        
        reduction_factor_1 = self.performance.get_reduction_factor(interventions, 1)
        reduction_factor_10 = self.performance.get_reduction_factor(interventions, 10)
        reduction_factor_14 = self.performance.get_reduction_factor(interventions, 14)
        reduction_factor_15 = self.performance.get_reduction_factor(interventions, 15)
        
        self.assertEqual(reduction_factor_1, 1)
        self.assertEqual(reduction_factor_10, 0.8)
        self.assertEqual(reduction_factor_14, 0.8)
        self.assertEqual(reduction_factor_15, 1)
    
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
