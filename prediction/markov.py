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
import numpy as np
from numpy.linalg import matrix_power
from scipy.optimize import minimize
from scipy.linalg import expm
from random import choices
from multiprocessing import Pool


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
        
    optimize_theta: 
        Optimize theta parameters
        
    initial_guess_theta:
        Initialize theta parameters
        
    get_mean_over_time:
        Get expected value over time
    
    get_std_over_time:
        Get standard deviation over time
        
    get_next_ic:
        Sample next index condition
        
    predict_mc:
        Monte Carlo simulation
        
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
        
        self._number_of_process = 1
        
        self._number_of_states = abs(worst_IC - best_IC) + 1
        self._is_transition_crescent = best_IC < worst_IC
        
        self.list_of_possible_ICs = np.arange(
            self.best_IC, self.worst_IC + 1, dtype=int
        )
        
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
            Time span for prediction (default 1)
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
            counts = np.unique(initial_vals, return_counts=True)
            n_transitions[counts[0] - 1] = counts[1]
        
        else:
            initial_vals = initial[temp == 1]
            counts = np.unique(initial_vals, return_counts=True)
            n_transitions[counts[0] - self.worst_IC] = counts[1]
        
        return n_transitions

    def _time_transitions(
        self, 
        initial: np.ndarray, 
        time: np.ndarray, 
        final: np.ndarray
    ) -> np.ndarray:
        """Get total time spent in each condition."""
        
        t_transitions = np.zeros(self._number_of_states)
        temp = initial - final
        
        if self._is_transition_crescent:
            relevant = temp <= 0
            times = time[relevant]
            indices = initial[relevant]
            
        else:
            relevant = temp >= 0
            times = time[relevant]
            indices = initial[relevant]
            
        for i in range(self.best_IC, self.worst_IC + 1):
            t_transitions[i - 1] += times[indices == i].sum()
            
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
        
        prob_matrix = np.array([self.transition_matrix_over_time(t) 
                                for t in range(max(time) + 1)])
                                
        prob_matrix_time = prob_matrix[time]
        
        if self._is_transition_crescent:
            prob_initial = prob_matrix_time[np.arange(len(initial)), initial - 1]
            likelihoods = prob_initial[np.arange(len(final)), final - 1]
        
        else:
            # Calculate likelihoods when transition matrix 
            # is ordered opposite
            pass
            # ...
        
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
        
        theta[-1] = 0
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
        
        if not(self._is_transition_crescent):
            self.theta = np.flip(self.theta)
        
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

    def get_next_IC(self, current_IC: int) -> int:
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

    def predict_MC(
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
        
        get_next_IC = self.get_next_IC  # Cache the method reference for better performance
        
        for t in range(1, delta_time):
            sample[t] = get_next_IC(sample[t-1])
        
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
        
        if self._number_of_process == 1:
            samples = [self.predict_MC(delta_time + 1,initial_IC) for _ in range(num_samples)]
            return np.mean(samples, axis=0)  # Mean pear year
        
        with Pool() as pool:
            results  = [pool.apply_async(self.predict_MC,
                                          (delta_time + 1, initial_IC))
                        for _ in range(num_samples)]
                        
            samples = [res.get() for res in results ]

        return np.mean(samples, axis=0)