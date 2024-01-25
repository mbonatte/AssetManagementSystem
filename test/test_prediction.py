import unittest
import random
import numpy as np

from ams.prediction.markov import MarkovContinous

class TestMarkovContinuous(unittest.TestCase):

    def setUp(self):
        # Common setup for all tests
        self.markov_crescent_0_5 = MarkovContinous(worst_IC=5, best_IC=0)
        self.markov_crescent_1_5 = MarkovContinous(worst_IC=5, best_IC=1)
        self.markov_crescent_2_5 = MarkovContinous(worst_IC=5, best_IC=2)
        
        self.markov_decrescent_5_0 = MarkovContinous(worst_IC=0, best_IC=5)
        self.markov_decrescent_5_1 = MarkovContinous(worst_IC=1, best_IC=5)
        self.markov_decrescent_5_2 = MarkovContinous(worst_IC=2, best_IC=5)
        
    def test_initialization(self):
        # Test initialization
        self.assertEqual(self.markov_crescent_1_5.worst_IC, 5)
        self.assertEqual(self.markov_crescent_1_5.best_IC, 1)
        
        self.assertEqual(self.markov_crescent_0_5._number_of_states, 6)
        self.assertEqual(self.markov_crescent_1_5._number_of_states, 5)
        self.assertEqual(self.markov_crescent_2_5._number_of_states, 4)
        
        self.assertEqual(self.markov_crescent_1_5._is_transition_crescent, True)
        
        
        self.assertEqual(self.markov_decrescent_5_1.worst_IC, 1)
        self.assertEqual(self.markov_decrescent_5_1.best_IC, 5)
        
        self.assertEqual(self.markov_decrescent_5_0._number_of_states, 6)
        self.assertEqual(self.markov_decrescent_5_1._number_of_states, 5)
        self.assertEqual(self.markov_decrescent_5_2._number_of_states, 4)
        
        self.assertEqual(self.markov_decrescent_5_1._is_transition_crescent, False)
        
    def test_theta_property(self):
        # Test getter and setter of theta
        test_theta_4 = np.array([0.5, 0.5, 0.5])
        test_theta_5 = np.array([0.5, 0.5, 0.5, 0.5])
        test_theta_6 = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        self.markov_crescent_0_5.theta = test_theta_6
        self.markov_crescent_1_5.theta = test_theta_5
        self.markov_crescent_2_5.theta = test_theta_4
        
        self.markov_decrescent_5_0.theta = test_theta_6
        self.markov_decrescent_5_1.theta = test_theta_5
        self.markov_decrescent_5_2.theta = test_theta_4
        
        np.testing.assert_array_equal(self.markov_crescent_0_5.theta, test_theta_6)
        np.testing.assert_array_equal(self.markov_crescent_1_5.theta, test_theta_5)
        np.testing.assert_array_equal(self.markov_crescent_2_5.theta, test_theta_4)
        
        np.testing.assert_array_equal(self.markov_decrescent_5_0.theta, test_theta_6)
        np.testing.assert_array_equal(self.markov_decrescent_5_1.theta, test_theta_5)
        np.testing.assert_array_equal(self.markov_decrescent_5_2.theta, test_theta_4)
        
    def test_intensity_matrix(self):
        self.markov_crescent_1_5.theta = np.array([1, 2, 3, 4])
        self.markov_decrescent_5_1.theta = np.array([1, 2, 3, 4])
        
        expected = np.array([[-1, 1 , 0 , 0, 0 ],
                             [0 , -2, 2 , 0, 0],
                             [0 , 0 , -3, 3, 0],
                             [0 , 0 , 0 , -4, 4],
                             [0 , 0 , 0 , 0 , 0]])
        
        np.testing.assert_array_almost_equal(self.markov_crescent_1_5.intensity_matrix, expected)
        np.testing.assert_array_almost_equal(self.markov_decrescent_5_1.intensity_matrix, expected)
    
    def test_transition_matrix(self):
        self.markov_crescent_1_5.theta = np.array([1, 2, 3, 4])
        self.markov_decrescent_5_1.theta = np.array([1, 2, 3, 4])
        
        expected = np.array([[0.36787944, 0.23254416, 0.14699594, 0.09291916, 0.1596613 ],
                             [0.        , 0.13533528, 0.17109643, 0.16223036, 0.53133793],
                             [0.        , 0.        , 0.04978707, 0.09441429, 0.85579864],
                             [0.        , 0.        , 0.        , 0.01831564, 0.98168436],
                             [0.        , 0.        , 0.        , 0.        , 1.      ]])
        
        np.testing.assert_array_almost_equal(self.markov_crescent_1_5.transition_matrix, expected)
        np.testing.assert_array_almost_equal(self.markov_decrescent_5_1.transition_matrix, expected)
        
    def test_transition_matrix_over_time(self):
        self.markov_crescent_1_5.theta = np.array([1, 2, 3, 4])
        self.markov_decrescent_5_1.theta = np.array([1, 2, 3, 4])
        
        expected = np.array([[6.73794700e-03, 6.69254707e-03, 6.64745304e-03, 6.60266286e-03, 9.73319390e-01],
                             [0.00000000e+00, 4.53999298e-05, 9.01880549e-05, 1.34370559e-04, 9.99730041e-01],
                             [0.00000000e+00, 0.00000000e+00, 3.05902321e-07, 9.11523501e-07, 9.99998783e-01],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.06115362e-09, 9.99999998e-01],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        
        np.testing.assert_array_almost_equal(self.markov_crescent_1_5.transition_matrix_over_time(5), expected)
        np.testing.assert_array_almost_equal(self.markov_decrescent_5_1.transition_matrix_over_time(5), expected)
        
    def test_crescent_number_transitions(self):
        initial_0_5 = np.array([0, 0, 1, 2, 3, 4, 5]*1000)
        final_0_5   = np.array([0, 1, 2, 3, 4, 5, 5]*1000)
    
        initial_1_5 = np.array([1, 1, 2, 3, 4, 5, 2]*1000)
        final_1_5   = np.array([1, 2, 3, 4, 5, 5, 1]*1000)
        
        initial_2_5 = np.array([2, 2, 3, 4, 5]*1000)
        final_2_5   = np.array([2, 3, 4, 5, 5]*1000)
        
        number_transitions_0_5 = self.markov_crescent_0_5._number_transitions(initial_0_5, final_0_5)
        number_transitions_1_5 = self.markov_crescent_1_5._number_transitions(initial_1_5, final_1_5)
        number_transitions_2_5 = self.markov_crescent_2_5._number_transitions(initial_2_5, final_2_5)
        
        np.testing.assert_array_equal(number_transitions_0_5, [1000, 1000, 1000, 1000, 1000,    0])
        np.testing.assert_array_equal(number_transitions_1_5, [1000, 1000, 1000, 1000,    0])
        np.testing.assert_array_equal(number_transitions_2_5, [1000, 1000, 1000,    0])
    
    def test_decrescent_number_transitions(self):
        initial_5_0 = np.array([5, 5, 4, 3, 2, 1, 0]*1000)
        final_5_0   = np.array([5, 4, 3, 2, 1, 0, 0]*1000)
        
        initial_5_1 = np.array([5, 5, 4, 3, 2, 1, 4]*1000)
        final_5_1   = np.array([5, 4, 3, 2, 1, 1, 5]*1000)
        
        initial_5_2 = np.array([5, 5, 4, 3, 2]*1000)
        final_5_2   = np.array([5, 4, 3, 2, 2]*1000)
        
        number_transitions_5_0 = self.markov_decrescent_5_0._number_transitions(initial_5_0, final_5_0)
        number_transitions_5_1 = self.markov_decrescent_5_1._number_transitions(initial_5_1, final_5_1)
        number_transitions_5_2 = self.markov_decrescent_5_2._number_transitions(initial_5_2, final_5_2)
        
        np.testing.assert_array_equal(number_transitions_5_0, [1000, 1000, 1000, 1000, 1000,    0])
        np.testing.assert_array_equal(number_transitions_5_1, [1000, 1000, 1000, 1000,    0])
        np.testing.assert_array_equal(number_transitions_5_2, [1000, 1000, 1000,    0])    

    def test_crescent_time_transitions(self):
        initial_0_5 = np.array([0, 0, 1, 2, 3, 4, 5]*1000)
        time_0_5    = np.array([1, 2, 3, 4, 5, 5, 5]*1000)
        final_0_5   = np.array([0, 1, 2, 3, 4, 5, 5]*1000)
    
        initial_1_5 = np.array([1, 1, 2, 3, 4, 5, 2]*1000)
        time_1_5    = np.array([1, 2, 3, 4, 5, 5, 5]*1000)
        final_1_5   = np.array([1, 2, 3, 4, 5, 5, 1]*1000)
        
        initial_2_5 = np.array([2, 2, 3, 4, 5]*1000)
        time_2_5    = np.array([1, 2, 3, 4, 5]*1000)
        final_2_5   = np.array([2, 3, 4, 5, 5]*1000)
        
        time_transitions_0_5 = self.markov_crescent_0_5._time_transitions(initial_0_5, time_0_5, final_0_5)
        time_transitions_1_5 = self.markov_crescent_1_5._time_transitions(initial_1_5, time_1_5, final_1_5)
        time_transitions_2_5 = self.markov_crescent_2_5._time_transitions(initial_2_5, time_2_5, final_2_5)
        
        np.testing.assert_array_equal(time_transitions_0_5, [3000, 3000, 4000, 5000, 5000, 5000])
        np.testing.assert_array_equal(time_transitions_1_5, [3000, 3000, 4000, 5000, 5000])
        np.testing.assert_array_equal(time_transitions_2_5, [3000, 3000, 4000, 5000])
    
    def test_decrescent_time_transitions(self):
        initial_5_0 = np.array([5, 5, 4, 3, 2, 1, 0]*1000)
        time_5_0    = np.array([1, 2, 3, 4, 5, 5, 5]*1000)
        final_5_0   = np.array([5, 4, 3, 2, 1, 0, 0]*1000)
        
        initial_5_1 = np.array([5, 5, 4, 3, 2, 1, 4]*1000)
        time_5_1    = np.array([1, 2, 3, 4, 5, 5, 5]*1000)
        final_5_1   = np.array([5, 4, 3, 2, 1, 1, 5]*1000)
        
        initial_5_2 = np.array([5, 5, 4, 3, 2]*1000)
        time_5_2    = np.array([1, 2, 3, 4, 5]*1000)
        final_5_2   = np.array([5, 4, 3, 2, 2]*1000)
        
        number_transitions_5_0 = self.markov_decrescent_5_0._time_transitions(initial_5_0, time_5_0, final_5_0)
        number_transitions_5_1 = self.markov_decrescent_5_1._time_transitions(initial_5_1, time_5_1, final_5_1)
        number_transitions_5_2 = self.markov_decrescent_5_2._time_transitions(initial_5_2, time_5_2, final_5_2)
        
        np.testing.assert_array_equal(number_transitions_5_0, [3000, 3000, 4000, 5000, 5000, 5000])
        np.testing.assert_array_equal(number_transitions_5_1, [3000, 3000, 4000, 5000, 5000])
        np.testing.assert_array_equal(number_transitions_5_2, [3000, 3000, 4000, 5000])    
    
    def test_crescent_initial_guess_theta(self):
        initial_0_5 = np.array([0, 0, 1, 2, 3, 4, 5]*1000)
        time_0_5    = np.array([1, 2, 3, 4, 5, 5, 5]*1000)
        final_0_5   = np.array([0, 1, 2, 3, 4, 5, 5]*1000)
    
        initial_1_5 = np.array([1, 1, 2, 3, 4, 5, 2]*1000)
        time_1_5    = np.array([1, 2, 3, 4, 5, 5, 5]*1000)
        final_1_5   = np.array([1, 2, 3, 4, 5, 5, 1]*1000)
        
        initial_2_5 = np.array([2, 2, 3, 4, 5]*1000)
        time_2_5    = np.array([1, 2, 3, 4, 5]*1000)
        final_2_5   = np.array([2, 3, 4, 5, 5]*1000)
        
        self.markov_crescent_0_5._initial_guess_theta(initial_0_5, time_0_5, final_0_5)
        self.markov_crescent_1_5._initial_guess_theta(initial_1_5, time_1_5, final_1_5)
        self.markov_crescent_2_5._initial_guess_theta(initial_2_5, time_2_5, final_2_5)
        
        np.testing.assert_array_almost_equal(self.markov_crescent_0_5.theta, [0.333333, 0.333333, 0.25    , 0.2     , 0.2     , 0.      ], decimal=6)
        np.testing.assert_array_almost_equal(self.markov_crescent_1_5.theta, [0.333333, 0.333333, 0.25    , 0.2     , 0.      ], decimal=6)
        np.testing.assert_array_almost_equal(self.markov_crescent_2_5.theta, [0.333333, 0.333333, 0.25    , 0.      ], decimal=6)
    
    def test_decrescent_initial_guess_theta(self):
        initial_5_0 = np.array([5, 5, 4, 3, 2, 1, 0]*1000)
        time_5_0    = np.array([1, 2, 3, 4, 5, 5, 5]*1000)
        final_5_0   = np.array([5, 4, 3, 2, 1, 0, 0]*1000)
        
        initial_5_1 = np.array([5, 5, 4, 3, 2, 1, 4]*1000)
        time_5_1    = np.array([1, 2, 3, 4, 5, 5, 5]*1000)
        final_5_1   = np.array([5, 4, 3, 2, 1, 1, 5]*1000)
        
        initial_5_2 = np.array([5, 5, 4, 3, 2]*1000)
        time_5_2    = np.array([1, 2, 3, 4, 5]*1000)
        final_5_2   = np.array([5, 4, 3, 2, 2]*1000)
        
        self.markov_decrescent_5_0._initial_guess_theta(initial_5_0, time_5_0, final_5_0)
        self.markov_decrescent_5_1._initial_guess_theta(initial_5_1, time_5_1, final_5_1)
        self.markov_decrescent_5_2._initial_guess_theta(initial_5_2, time_5_2, final_5_2)
        
        np.testing.assert_array_almost_equal(self.markov_decrescent_5_0.theta, [0.333333, 0.333333, 0.25    , 0.2     , 0.2     , 0.      ], decimal=6)
        np.testing.assert_array_almost_equal(self.markov_decrescent_5_1.theta, [0.333333, 0.333333, 0.25    , 0.2     , 0.      ], decimal=6)
        np.testing.assert_array_almost_equal(self.markov_decrescent_5_2.theta, [0.333333, 0.333333, 0.25    , 0.      ], decimal=6) 
    
    def test_crescent_likelihood(self):
        initial_0_5 = np.array([0,1,2,3,4,5])
        time_0_5 = np.array([5,1,2,3,4,5])
        final_0_5 = np.array([0,1,2,3,4,5])
        
        initial_1_5 = np.array([1,2,3,4,5])
        time_1_5 = np.array([1,2,3,4,5])
        final_1_5 = np.array([1,2,3,4,5])
        
        initial_2_5 = np.array([2,3,4,5])
        time_2_5 = np.array([2,3,4,5])
        final_2_5 = np.array([2,3,4,5])
        
        self.markov_crescent_0_5.theta = np.array([0.5, 1, 2, 4, 0.5])
        self.markov_crescent_1_5.theta = np.array([0.5, 1, 2, 4])
        self.markov_crescent_2_5.theta = np.array([0.5, 1, 2])
        
        self.assertAlmostEqual(self.markov_crescent_0_5.likelihood(initial_0_5, time_0_5, final_0_5), 21.5)
        self.assertAlmostEqual(self.markov_crescent_1_5.likelihood(initial_1_5, time_1_5, final_1_5), 24.5)
        self.assertAlmostEqual(self.markov_crescent_2_5.likelihood(initial_2_5, time_2_5, final_2_5), 12)
    
    def test_decrescent_likelihood(self):
        initial_5_0 = np.array([5,4,3,2,1,0])
        time_5_0 = np.array([5,1,2,3,4,5])
        final_5_0 = np.array([5,4,3,2,1,0])
        
        initial_5_1 = np.array([5,4,3,2,1])
        time_5_1 = np.array([1,2,3,4,5])
        final_5_1 = np.array([5,4,3,2,1])
        
        initial_5_2 = np.array([5,4,3,2])
        time_5_2 = np.array([2,3,4,5])
        final_5_2 = np.array([5,4,3,2])
        
        self.markov_decrescent_5_0.theta = np.array([0.5, 1, 2, 4, 0.5])
        self.markov_decrescent_5_1.theta = np.array([0.5, 1, 2, 4])
        self.markov_decrescent_5_2.theta = np.array([0.5, 1, 2])
        
        self.assertAlmostEqual(self.markov_decrescent_5_0.likelihood(initial_5_0, time_5_0, final_5_0), 21.5)
        self.assertAlmostEqual(self.markov_decrescent_5_1.likelihood(initial_5_1, time_5_1, final_5_1), 24.5)
        self.assertAlmostEqual(self.markov_decrescent_5_2.likelihood(initial_5_2, time_5_2, final_5_2), 12)
    
    def test_update_theta_call_likelihood(self):
        theta = np.array([0.5, 1, 2, 4])
        
        initial_1_5 = np.array([1,2,3,4,5])
        time_1_5 = np.array([1,2,3,4,5])
        final_1_5 = np.array([1,2,3,4,5])
        
        initial_5_1 = np.array([5,4,3,2,1])
        time_5_1 = np.array([1,2,3,4,5])
        final_5_1 = np.array([5,4,3,2,1])
        
        likelihood_1_5 = self.markov_crescent_1_5._update_theta_call_likelihood(theta, initial_1_5, time_1_5, final_1_5, [0])
        likelihood_5_1 = self.markov_decrescent_5_1._update_theta_call_likelihood(theta, initial_5_1, time_5_1, final_5_1, [0])
        
        self.assertAlmostEqual(likelihood_1_5, 24.5)
        self.assertAlmostEqual(likelihood_5_1, 24.5)
    
    def test_optimize_theta(self):
        initial_1_5 = np.array([1, 1, 2, 3, 4, 5]*1000)
        time_1_5    = np.array([1, 2, 3, 4, 5, 5]*1000)
        final_1_5   = np.array([1, 2, 3, 4, 5, 5]*1000)
        
        initial_5_1 = np.array([5, 5, 4, 3, 2, 1]*1000)
        time_5_1    = np.array([1, 2, 3, 4, 5, 5]*1000)
        final_5_1   = np.array([5, 4, 3, 2, 1, 1]*1000)
        
        
        self.markov_crescent_1_5._initial_guess_theta(initial_1_5, time_1_5, final_1_5)
        self.markov_crescent_1_5._optimize_theta(initial_1_5, time_1_5, final_1_5)
        
        self.markov_decrescent_5_1._initial_guess_theta(initial_5_1, time_5_1, final_5_1)
        self.markov_decrescent_5_1._optimize_theta(initial_5_1, time_5_1, final_5_1)
        
        expected = np.array([0.508275, 0.410478, 0.281638, 0.247331, 0.      ])
        
        np.testing.assert_array_almost_equal(self.markov_crescent_1_5.theta, expected, decimal=6)
        np.testing.assert_array_almost_equal(self.markov_decrescent_5_1.theta, expected, decimal=6)
    
    def test_crescent_fit(self):
        initial_0_5 = np.array([0, 0, 1, 2, 3, 4, 5]*1000)
        time_0_5    = np.array([5, 1, 2, 3, 4, 5, 5]*1000)
        final_0_5   = np.array([0, 1, 2, 3, 4, 5, 5]*1000)
    
        initial_1_5 = np.array([1, 1, 2, 3, 4, 5]*1000)
        time_1_5    = np.array([1, 2, 3, 4, 5, 5]*1000)
        final_1_5   = np.array([1, 2, 3, 4, 5, 5]*1000)
        
        initial_2_5 = np.array([2, 2, 3, 4, 5]*1000)
        time_2_5    = np.array([1, 2, 3, 4, 5]*1000)
        final_2_5   = np.array([2, 3, 4, 5, 5]*1000)
    
        self.markov_crescent_0_5.fit(initial_0_5, time_0_5, final_0_5)
        self.markov_crescent_1_5.fit(initial_1_5, time_1_5, final_1_5)
        self.markov_crescent_2_5.fit(initial_2_5, time_2_5, final_2_5)
        
        expected_0_5 = np.array([0.180274, 0.74853 , 0.394773, 0.282769, 0.247219, 0.      ])
        expected_1_5 = np.array([0.508275, 0.410478, 0.281638, 0.247331, 0.      ])
        expected_2_5 = np.array([0.508785, 0.405091, 0.31704 , 0.      ])
        
        np.testing.assert_array_almost_equal(self.markov_crescent_0_5.theta, expected_0_5, decimal=6)
        np.testing.assert_array_almost_equal(self.markov_crescent_1_5.theta, expected_1_5, decimal=6)
        np.testing.assert_array_almost_equal(self.markov_crescent_2_5.theta, expected_2_5, decimal=6)
    
    def test_decrescent_fit(self):
        initial_5_0 = np.array([5, 5, 4, 3, 2, 1, 0]*1000)
        time_5_0    = np.array([5, 1, 2, 3, 4, 5, 5]*1000)
        final_5_0   = np.array([5, 4, 3, 2, 1, 0, 0]*1000)
        
        initial_5_1 = np.array([5, 5, 4, 3, 2, 1]*1000)
        time_5_1    = np.array([1, 2, 3, 4, 5, 5]*1000)
        final_5_1   = np.array([5, 4, 3, 2, 1, 1]*1000)
        
        initial_5_2 = np.array([5, 5, 4, 3, 2]*1000)
        time_5_2    = np.array([1, 2, 3, 4, 5]*1000)
        final_5_2   = np.array([5, 4, 3, 2, 2]*1000)
        
        self.markov_decrescent_5_0.fit(initial_5_0, time_5_0, final_5_0)
        self.markov_decrescent_5_1.fit(initial_5_1, time_5_1, final_5_1)
        self.markov_decrescent_5_2.fit(initial_5_2, time_5_2, final_5_2)
        
        expected_5_0 = np.array([0.180274, 0.74853 , 0.394773, 0.282769, 0.247219, 0.      ])
        expected_5_1 = np.array([0.508275, 0.410478, 0.281638, 0.247331, 0.      ])
        expected_5_2 = np.array([0.508785, 0.405091, 0.31704 , 0.      ])
        
        np.testing.assert_array_almost_equal(self.markov_decrescent_5_0.theta, expected_5_0, decimal=6)
        np.testing.assert_array_almost_equal(self.markov_decrescent_5_1.theta, expected_5_1, decimal=6)
        np.testing.assert_array_almost_equal(self.markov_decrescent_5_2.theta, expected_5_2, decimal=6)
        
    def test_mean_prediction(self):
        self.markov_crescent_1_5.theta   = np.array([0.508275, 0.410478, 0.281638, 0.247331, 0.      ])
        self.markov_decrescent_5_1.theta = np.array([0.508275, 0.410478, 0.281638, 0.247331, 0.      ])
        
        mean_1_5 = self.markov_crescent_1_5.get_mean_over_time(delta_time=10, initial_IC=2)
        mean_5_1 = self.markov_decrescent_5_1.get_mean_over_time(delta_time=10, initial_IC=4)
        
        diff_to_worst = 0.592958833471442
        
        self.assertAlmostEqual(mean_1_5[-1], self.markov_crescent_1_5.worst_IC - diff_to_worst, places=4)
        self.assertAlmostEqual(mean_5_1[-1], self.markov_decrescent_5_1.worst_IC + diff_to_worst, places=4)
    
    def test_std_prediction(self):
        self.markov_crescent_1_5.theta   = np.array([0.508275, 0.410478, 0.281638, 0.247331, 0.      ])
        self.markov_decrescent_5_1.theta = np.array([0.508275, 0.410478, 0.281638, 0.247331, 0.      ])
        
        std_1_5 = self.markov_crescent_1_5.get_std_over_time(delta_time=10, initial_IC=2)
        std_5_1 = self.markov_decrescent_5_1.get_std_over_time(delta_time=10, initial_IC=4)
        
        self.assertAlmostEqual(std_1_5[-1], 0.7851147079718335, places=4)
        self.assertAlmostEqual(std_5_1[-1], 0.7851147079718335, places=4)
    
    def test_next_state_sampling(self):
        self.markov_crescent_1_5.theta   = np.array([0.5, 1, 1.5, 2])
        self.markov_decrescent_5_1.theta = np.array([0.5, 1, 1.5, 2])
        
        current_IC = 3
        num_samples = 200
        
        counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        random.seed(1)
        for i in range(num_samples):
            next_state = self.markov_crescent_1_5._get_next_IC(current_IC)
            counts[next_state] += 1
        probs_1_5 = np.array([counts[i]/num_samples for i in [1,2,3,4,5]])
        
        counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        random.seed(1)
        for i in range(num_samples):
            next_state = self.markov_decrescent_5_1._get_next_IC(current_IC)
            counts[next_state] += 1
        probs_5_1 = np.array([counts[i]/num_samples for i in [1,2,3,4,5]])
        
        expected_probs_1_5 = self.markov_crescent_1_5.transition_matrix[abs(current_IC-self.markov_crescent_1_5.best_IC)]
        expected_probs_5_1 = self.markov_decrescent_5_1.transition_matrix[abs(current_IC-self.markov_crescent_1_5.best_IC)]
        
        np.testing.assert_array_almost_equal(probs_1_5, expected_probs_1_5, decimal=2)
        np.testing.assert_array_almost_equal(np.flip(probs_5_1), expected_probs_5_1, decimal=2)
        
    def test_mc_prediction(self):
        # Test if MC prediction is close to analytical
        delta_time = 10
        
        self.markov_crescent_1_5.theta   = np.array([0.508275, 0.410478, 0.281638, 0.247331, 0.      ])
        self.markov_decrescent_5_1.theta = np.array([0.508275, 0.410478, 0.281638, 0.247331, 0.      ])
        
        analytical_1_5 = self.markov_crescent_1_5.get_mean_over_time(delta_time, 2)[-1]
        analytical_5_1 = self.markov_crescent_1_5.get_mean_over_time(delta_time, 4)[-1]
        
        random.seed(2)
        mc_1_5 = self.markov_crescent_1_5.get_mean_over_time_MC(delta_time, 2, num_samples=20)[-1]
        random.seed(2)
        mc_5_1 = self.markov_crescent_1_5.get_mean_over_time_MC(delta_time, 4, num_samples=40)[-1]
        
        self.assertLess(abs(analytical_1_5 - mc_1_5), 0.01)
        self.assertLess(abs(analytical_5_1 - mc_5_1), 0.01)
        
if __name__ == '__main__':
    unittest.main()
