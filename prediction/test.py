import unittest
from markov import MarkovContinous
import numpy as np

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
        self.markov.fit(self.initial, self.time, self.final)
        
        mean = self.markov.get_mean_over_time(delta_time=10, initial_IC=2)
        self.assertAlmostEqual(mean[-1], 4.407041166528558, places=4)
        
    def test_next_state_sampling(self):
        self.markov.theta = np.array([0.5, 1, 1.5, 2])
        
        counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        num_samples = 10000
        
        for i in range(num_samples):
            next_state = self.markov.get_next_IC(current_IC=3)
            counts[next_state] += 1
            
        probs = np.array([counts[i]/num_samples for i in [1,2,3,4,5]])
        expected_probs = self.markov.transition_matrix[2]
        np.testing.assert_array_almost_equal(probs, expected_probs, decimal=2)    
        
    def test_mc_prediction(self):
        # Test MC prediction is close to analytical
        self.markov.fit(self.initial, self.time, self.final)
        
        analytical = self.markov.get_mean_over_time(delta_time=10, initial_IC=2)[-1]
        mc = self.markov.get_mean_over_time_MC(delta_time=10, initial_IC=2, num_samples=10000)[-1]
        
        self.assertLess(abs(analytical - mc), 0.02)
        
if __name__ == '__main__':
    unittest.main()