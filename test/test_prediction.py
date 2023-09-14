import unittest
import random
import numpy as np

from prediction.markov import MarkovContinous

class TestMarkovContinuous(unittest.TestCase):

    def setUp(self):
        self.markov = MarkovContinous(worst_IC=5, best_IC=1)
        self.initial = np.array([1, 1, 2, 3, 4, 5]*1000)
        self.time = np.array([1, 2, 3, 4, 5, 5]*1000)
        self.final = np.array([1, 2, 3, 4, 5, 5]*1000)
        
    def test_transition_matrix(self):
        self.markov.theta = np.array([1, 2, 3, 4]) 
        expected = np.array([[0.36787944, 0.23254416, 0.14699594, 0.09291916, 0.1596613 ],
                             [0.        , 0.13533528, 0.17109643, 0.16223036, 0.53133793],
                             [0.        , 0.        , 0.04978707, 0.09441429, 0.85579864],
                             [0.        , 0.        , 0.        , 0.01831564, 0.98168436],
                             [0.        , 0.        , 0.        , 0.        , 1.      ]])
        np.testing.assert_array_almost_equal(self.markov.transition_matrix, expected)
        
    def test_transition_matrix_over_time(self):
        self.markov.theta = np.array([1, 2, 3, 4]) 
        expected = np.array([[6.73794700e-03, 6.69254707e-03, 6.64745304e-03, 6.60266286e-03, 9.73319390e-01],
                             [0.00000000e+00, 4.53999298e-05, 9.01880549e-05, 1.34370559e-04, 9.99730041e-01],
                             [0.00000000e+00, 0.00000000e+00, 3.05902321e-07, 9.11523501e-07, 9.99998783e-01],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.06115362e-09, 9.99999998e-01],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        np.testing.assert_array_almost_equal(self.markov.transition_matrix_over_time(5), expected)
        
    def test_likelihood(self):
        initial = np.array([1,2,3,4,5])
        time = np.array([1,2,3,4,5])
        final = np.array([1,2,3,4,5])
        
        self.markov.theta = np.array([0.5, 1, 2, 4])
        
        log_lik = self.markov.likelihood(initial, time, final)
        self.assertAlmostEqual(log_lik, 24.5)
        
    def test_optimize_theta(self):
        self.markov.fit(self.initial, self.time, self.final)
        
        expected = np.array([0.508275, 0.410478, 0.281638, 0.247331, 0.      ])
        np.testing.assert_array_almost_equal(self.markov.theta, expected, decimal=6)
    
    def test_mean_prediction(self):
        self.markov.theta = np.array([0.508275, 0.410478, 0.281638, 0.247331, 0.      ])
        
        mean = self.markov.get_mean_over_time(delta_time=10, initial_IC=2)
        self.assertAlmostEqual(mean[-1], 4.407041166528558, places=4)
    
    def test_std_prediction(self):
        self.markov.theta = np.array([0.508275, 0.410478, 0.281638, 0.247331, 0.      ])
        
        mean = self.markov.get_std_over_time(delta_time=10, initial_IC=2)
        self.assertAlmostEqual(mean[-1], 0.7851147079718335, places=4)
    
    def test_next_state_sampling(self):
        current_IC = 3
        self.markov.theta = np.array([0.5, 1, 1.5, 2])
        
        counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        num_samples = 200
        
        random.seed(1)
        for i in range(num_samples):
            next_state = self.markov.get_next_IC(current_IC)
            counts[next_state] += 1
            
        probs = np.array([counts[i]/num_samples for i in [1,2,3,4,5]])
        expected_probs = self.markov.transition_matrix[current_IC-1]
        np.testing.assert_array_almost_equal(probs, expected_probs, decimal=2)
        
    def test_mc_prediction(self):
        # Test if MC prediction is close to analytical
        delta_time = 10
        initial_IC = 2
        
        self.markov.fit(self.initial, self.time, self.final)
        
        analytical = self.markov.get_mean_over_time(delta_time, initial_IC)[-1]
        
        random.seed(2)
        self.markov._number_of_process = 1
        mc = self.markov.get_mean_over_time_MC(delta_time, initial_IC, num_samples=20)[-1]
        
        self.assertLess(abs(analytical - mc), 0.01)
        
if __name__ == '__main__':
    unittest.main()
