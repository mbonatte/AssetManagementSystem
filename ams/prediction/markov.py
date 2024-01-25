"""

Created on Sep 21, 2022.

@author: MauricioBonatte
@e-mail: mbonatte@ymail.com

########

Markov continuous model.

Continuous stochastic process where the process changes state based on 
exponential distribution and transitions specified by a stochastic matrix.

"""

from typing import List, Tuple
from random import choices
from multiprocessing import Pool

import numpy as np

from numpy.linalg import matrix_power
from scipy.optimize import minimize
from scipy.linalg import expm

class MarkovContinous():
    """
    Markov continuous model class.

    Attributes
    ----------
    worst_IC : int
        Worst condition the asset can reach
    
    best_IC : int
        Best condition the asset can reach
    
    optimizer : bool, default=True
        Whether to optimize the model
    
    verbose : bool, default=False
        Whether to print optimization results

    Methods
    -------
    fit(initial, time, final):
        Fit model to data
        
    likelihood:
        Calculate likelihood
        
    get_mean_over_time:
        Get expected value over time
    
    get_std_over_time:
        Get standard deviation over time
        
    get_mean_over_time_mc:
        Get expected value via Monte Carlo
    """

    def __init__(
        self,
        worst_IC: int,
        best_IC: int,
        optimizer: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the MarkovContinuous model.

        Parameters
        ----------
        worst_IC : int
            the worst condition the asse can reach
        best_IC : int
            the best condition the asse can reach
        optimizer : bool
            to optize the model or not
        verbose : bool
            print the results through the execution
        """
        self.worst_IC = worst_IC
        self.best_IC = best_IC
        self.verbose = verbose
        self.optimizer = optimizer
        
        self._number_of_states = abs(worst_IC - best_IC) + 1
        self._is_transition_crescent = best_IC < worst_IC
        
        self.list_of_possible_ICs = np.linspace(start = self.best_IC,
                                                stop = self.worst_IC,
                                                num = self._number_of_states,
                                                dtype = int)
        
        self._is_fitted = False

        if not verbose:
            np.seterr(all='ignore')

    @property
    def theta(self) -> np.ndarray:
        """Return theta parameters if model is fitted."""
        
        if not self._is_fitted:
            raise RuntimeError('Markov model is not fitted')
            
        return self._theta

    @theta.setter
    def theta(self, new_theta: np.ndarray) -> None:
        """Set theta parameters and cache intensity and transition matrices."""
        
        self._theta = new_theta
        self._is_fitted = True
        self._set_intensity_matrix()
        self._set_transition_matrix()

    @property
    def intensity_matrix(self) -> np.matrix:
        """Return intensity matrix."""
        
        return self._intensity_matrix

    def _set_intensity_matrix(self) -> None:
        """Set intensity matrix."""
        
        self._intensity_matrix = np.zeros((self._number_of_states,
                                            self._number_of_states))
        
        for i in range(self._number_of_states - 1):
            self._intensity_matrix[i, i] = -self.theta[i]
            self._intensity_matrix[i, i + 1] = self.theta[i]

    @property
    def transition_matrix(self) -> np.matrix:
        """Return transition matrix."""
        
        return self._transition_matrix

    def _set_transition_matrix(self) -> None:
        """Set transition matrix."""
        
        self._transition_matrix = expm(self.intensity_matrix)

    def transition_matrix_over_time(
        self, delta_time: int = 1
    ) -> np.matrix:
        """
        Return transition matrix raised to the power of time.

        Parameters
        ----------
        delta_time : int, optional
            Time span for prediction (default 1 unit time)
        """
        
        return matrix_power(self.transition_matrix,
                            delta_time)

    def _number_transitions(
        self, initial: np.ndarray, final: np.ndarray
    ) -> np.ndarray:
        """Get number of transitions between conditions."""
        
        n_transitions = np.zeros(self._number_of_states)
        temp = initial - final
        
        if self._is_transition_crescent:
            initial_vals = initial[temp == -1]
        else:
            initial_vals = initial[temp == 1]
            
        counts = np.unique(initial_vals, return_counts=True)
            
        n_transitions[abs(counts[0] - self.best_IC)] = counts[1]
        
        return n_transitions

    def _time_transitions(
        self, 
        initial: np.ndarray, 
        time: np.ndarray, 
        final: np.ndarray
    ) -> np.ndarray:
        """Get total time spent in each condition."""
        
        # Determine relevant transitions based on crescent or decrescent nature.
        if self._is_transition_crescent:
            relevant = initial <= final
        else:
            relevant = initial >= final
            
        # Filter the relevant times and initial states.
        times = time[relevant]
        indices = initial[relevant]
        
        # Adjust indices based on crescent or decrescent nature.
        adjusted_indices = abs(indices - self.best_IC)

        # np.bincount handles the summation for each unique index in adjusted_indices.
        # The minlength parameter ensures the output array has the desired length.
        t_transitions = np.bincount(adjusted_indices, weights=times, minlength=self._number_of_states)
        
        return t_transitions

    def likelihood(
        self, 
        initial: np.ndarray,
        time: np.ndarray, 
        final: np.ndarray
    ) -> float:
        """
        Calculate likelihood of model given data.
        
        Parameters
        ----------
        initial : np.array
            Initial condition indices
            
        time : np.array
            Timesteps between conditions
        
        final : np.array
            Final condition indices
            
        Returns
        -------
        float
            Log-likelihood
        """
        # Create an array where each entry corresponds to the probability of transitioning
        # from one state to another over each possible time step.
        prob_matrix = np.array([self.transition_matrix_over_time(t) 
                                for t in range(max(time) + 1)])
        
        # Select the relevant transition probabilities for each observed transition,
        # based on the actual time elapsed for each transition.
        prob_matrix_time = prob_matrix[time]
        
        # Calculate the probabilities of the observed transitions.
        prob_initial = prob_matrix_time[np.arange(len(initial)), abs(initial - self.best_IC)]
        likelihoods = prob_initial[np.arange(len(final)), abs(final - self.best_IC)]
                
        log_likelihoods = np.log(likelihoods)
        
        return -log_likelihoods.sum()

    def _update_theta_call_likelihood(
        self,
        theta: np.ndarray,
        initial: np.ndarray,
        time: np.ndarray, 
        final: np.ndarray,
        n_iter: List[int]
    ) -> float:
        """
        Update theta and return likelihood.
        
        Called by optimizer.

        Parameters
        ----------
        theta : np.array
            DESCRIPTION.
        initial : np.array
            Initial condition indices
        time : np.array
            Timesteps between conditions
        final : np.array
            Final condition indices
            Current iteration step.
            
        Returns
        -------
        float
            Log-likelihood
        """
        
        n_iter[0] += 1
        self.theta = theta
        
        likelihood = self.likelihood(initial, time, final)
        
        if self.verbose:
            print(f"Iteration {n_iter[0]}, likelihood: {likelihood}")
            
        return likelihood

    def _optimize_theta(
        self,
        initial: np.ndarray,
        time: np.ndarray,
        final: np.ndarray    
    ) -> None:
        """Optimize theta parameters to maximize likelihood.

        Parameters
        ----------
        initial : np.array
            Initial condition indices
        time : np.array
            Timesteps between conditions
        final : np.array
            Final condition indices
            Current iteration step.

        Returns
        -------
        None.
        """
        n_ite = [0]
        bounds = [(0, None) for n in range(self._number_of_states)]
        
        minimize(
            fun=self._update_theta_call_likelihood,
            x0=self.theta,
            args=(initial, time, final, n_ite,),
            method='SLSQP',  # why was this specific algorithm selected?
            bounds=bounds,
            options={'disp': self.verbose}
        )
        
        self._is_fitted = True

    def _initial_guess_theta(
        self, 
        initial: np.ndarray,
        time: np.ndarray,
        final: np.ndarray
    ) -> None:
        """
        Initialize theta parameters.

        Parameters
        ----------
        initial : np.array
            Initial condition indices
        time : np.array
            Timesteps between conditions
        final : np.array
            Final condition indices
            Current iteration step.

        Returns
        -------
        None.
        """
        
        n_transitions = self._number_transitions(initial, final)
        t_transitions = self._time_transitions(initial, time, final)
        
        self.theta = n_transitions/t_transitions
        self.theta = np.where(np.isnan(self.theta), 30, self.theta)
        
        self._is_fitted = True

    def fit(
        self,
        initial: np.ndarray,
        time: np.ndarray,
        final: np.ndarray
    ) -> None:
        """
        Fit model to data by optimizing theta.

        Parameters
        ----------
        initial : np.array
            Initial condition indices
        time : np.array
            Timesteps between conditions
        final : np.array
            Final condition indices
            Current iteration step.

        Returns
        -------
        None.

        """
        
        initial, time, final = map(np.array, (initial, time, final))
        
        self._initial_guess_theta(initial, time, final)
        
        if self.verbose:
            print('prior likelihood = ', self.likelihood(initial, time, final))
        
        if self.optimizer:
            self._optimize_theta(initial, time, final)
        
        if self.verbose:
            print("Posterior likelihood:", self.likelihood(initial, time, final))

    def get_mean_over_time(
        self, delta_time: int, initial_IC: int = None
    ) -> np.ndarray:
        """
        Get analytically expected IC over time.
        
        Parameters
        ----------
        delta_time : int
            Timesteps to calculate 
            
        initial_IC : int, optional
            Initial condition index
        """
        
        if initial_IC is None:
            initial_IC = self.best_IC
            
        initial_IC_index = abs(initial_IC - self.best_IC)
        
        mean_IC = np.empty(delta_time + 1)
        
        for t in range(delta_time + 1):
            prob = self.transition_matrix_over_time(t)[initial_IC_index]
            mean_IC[t] = prob.dot(self.list_of_possible_ICs)
            
        return mean_IC

    def get_std_over_time(
        self, delta_time: int, initial_IC: int = None
    ) -> np.ndarray:
        """
        Get analytically standard deviation over time.
        
        Parameters
        ----------
        delta_time : int
            Timesteps to calculate
            
        initial_IC : int, optional
            Initial condition index
        """
        
        if initial_IC is None:
            initial_IC = self.best_IC
            
        initial_IC_index = abs(initial_IC - self.best_IC)
        
        std_IC = np.empty(delta_time + 1)

        for t in range(0, delta_time + 1):
        
            prob = self.transition_matrix_over_time(t)[initial_IC_index]
            mean = prob.dot(self.list_of_possible_ICs)
            
            vari = prob.dot((self.list_of_possible_ICs - mean) ** 2)
            std_IC[t] = vari ** 0.5
            
        return std_IC

    def _get_next_IC(self, current_IC: int) -> int:
        """
        Sample next index condition.

        Parameters
        ----------
        current_IC : int
            Current condition index

        Returns
        -------
        int
            Next sampled condition index

        """
        
        IC_index = abs(current_IC-self.best_IC)
        prob = self.transition_matrix[IC_index]
        
        return choices(self.list_of_possible_ICs, weights=prob, k=1)[0]

    def _predict_MC(
        self, delta_time: int, initial_IC: int
    ) -> np.ndarray:
        """
        Monte Carlo simulation of condition indices.
        
        Parameters
        ----------
        delta_time : int
            Timesteps to simulate
            
        initial_IC : int
            Initial condition index
            
        Returns
        -------
        np.array
            Simulated condition index trajectory  
        """
        
        sample = np.empty(delta_time, dtype=int)
        sample[0] = initial_IC
        
        _get_next_IC = self._get_next_IC  # Cache the method reference for better performance
        
        for t in range(1, delta_time):
            sample[t] = _get_next_IC(sample[t-1])
        
        return sample

    def get_mean_over_time_MC(
        self, 
        delta_time: int,
        initial_IC: int = None,
        num_samples: int = 1000
    ) -> np.ndarray:
        """
        Get mean prediction via Monte Carlo simulation.
        
        Parameters
        ----------
        delta_time : int
            Timesteps to simulate
            
        initial_IC : int, optional
            Initial condition index
            
        num_samples : int, optional
            Number of simulations to run
            
        Returns
        -------
        np.array
            Mean prediction over time
        """
        
        if initial_IC is None:
            initial_IC = self.best_IC
        
        samples = [self._predict_MC(delta_time + 1,initial_IC) for _ in range(num_samples)]
        return np.mean(samples, axis=0)  # Mean pear year