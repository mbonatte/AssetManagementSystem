"""
Created on Sep 21, 2022.

@author: MauricioBonatte
@e-mail: mbonatte@ymail.com
"""

import numpy as np
from numpy.linalg import matrix_power
from scipy.optimize import minimize
from scipy.linalg import expm
from random import choices
from multiprocessing import Pool


class MarkovContinous():
    """
    The Markov continous model.

    Continuous stochastic process in which, for each state, or index condition
    (IC), the process will change state according to an exponential random
    variable and then move to a different state as specified by the probabili-
    ties of a stochastic matrix.

    Attributes
    ----------
    worst_IC : int
        the worst condition the asset can reach
    best_IC : int
        the best condition the asset can reach
    optimizer : bool, default=True
        to optize the model or not
    verbose : bool, default=False
        print the results through the execution

    Methods
    -------
    fit(initial, time, final)
        Fit the model using the input data
    """

    def __init__(self,
                 worst_IC: int,
                 best_IC: int,
                 optimizer=True,
                 verbose=False):
        """
        Construct the class.

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
        self._is_transition_crescent = True if (best_IC < worst_IC) else False
        self.list_of_possible_ICs = np.linspace(start=self.best_IC,
                                                stop=self.worst_IC,
                                                num=self._number_of_states,
                                                dtype=int)
        self._is_fitted = False

        if verbose is False:
            np.seterr(all='ignore')

    @property
    def theta(self):
        """Return theta if model is fitted."""
        if not self._is_fitted:
            raise RuntimeError('Markov model is not fitted')
        return self._theta

    @theta.setter
    def theta(self, new_theta):
        """Set theta and save intensity and transition matrices in cache."""
        self._theta = new_theta
        self._is_fitted = True
        self.__set_intensity_matrix()
        self.__set_transition_matrix()

    @property
    def intensity_matrix(self) -> np.matrix:
        """Return the intensity matrix."""
        return self.__intensity_matrix

    def __set_intensity_matrix(self):
        """Set the intensity matrix."""
        self.__intensity_matrix = np.zeros((self._number_of_states,
                                            self._number_of_states))
        for i in range(self._number_of_states-1):
            self.__intensity_matrix[i][i] = -self.theta[i]
            self.__intensity_matrix[i][i+1] = self.theta[i]

    @property
    def transition_matrix(self) -> np.matrix:
        """Return the transition matrix."""
        return self.__transition_matrix

    def __set_transition_matrix(self):
        """Set the transition matrix."""
        self.__transition_matrix = expm(self.intensity_matrix)

    def transition_matrix_over_time(self, delta_time: int = 1) -> np.matrix:
        """
        Return the transition matrix raised to the power of time.

        If the argument `delta_time` isn't passed in, the default time
        considered is 1 unit time.

        Parameters
        ----------
        delta_time : int, optional
            The time span for the prediction (default is 1 unit time)
        """
        return matrix_power(self.transition_matrix,
                            delta_time)

    def _number_transitions(self, initial, final) -> np.array:
        """Get number of Transitions from one Conditon to the next one."""
        n_transitions = np.zeros(self._number_of_states)
        temp = initial - final
        if self._is_transition_crescent:
            _initial = initial[temp == -1]
            total = np.unique(_initial, return_counts=True)
            for ini, n in zip(total[0], total[1]):
                n_transitions[ini-1] = n
        else:
            _initial = initial[temp == 1]
            total = np.unique(_initial, return_counts=True)
            for ini, n in zip(total[0], total[1]):
                n_transitions[ini-self.worst_IC] = n
        return n_transitions

    def _time_transitions(self, initial, time, final):
        """Sum of all time the asset stayed in the same Conditon State."""
        t_transitions = np.zeros(self._number_of_states)
        temp = initial - final
        if self._is_transition_crescent:
            _time = time[temp <= 0]
            _initial = initial[temp <= 0]
            for i in range(self.best_IC, self.worst_IC+1, 1):
                t_transitions[i-1] += sum(_time[_initial == i])
        else:
            _time = time[temp >= 0]
            _initial = initial[temp >= 0]
            for i in range(self.best_IC, self.worst_IC+1, 1):
                t_transitions[i-self.worst_IC] += sum(_time[_initial == i])
        # I don't remember why I commented it, but it must be only used when
        # dealing with smal data sample. I Need to verify it further.
        # To not have a time transition equal to 0.
        # t_transitions = np.clip(t_transitions, 1,None)
        return t_transitions

    def likelihood(self, initial: np.array,
                   time: np.array, final: np.array) -> float:
        """
        Calculate the likelihood of the model.

        Parameters
        ----------
        initial : np.array
            Initial condition index.
        time : np.array
            Time from one condition index to another.
        final : np.array
            Final condition index.

        Returns
        -------
        float
            The log-likelihood of the model.

        """
        prob_matrix = np.array([self.transition_matrix_over_time(t)
                                for t in range(max(time)+1)])
        prob_matrix_time = prob_matrix[time]
        if self._is_transition_crescent:
            prob_initial = np.array(
                [t[i] for t, i in zip(prob_matrix_time, initial-1)])
            likelihoods = np.array([t[i]
                                    for t, i in zip(prob_initial, final-1)])
        log_likelihoods = np.log(likelihoods)
        return -sum(log_likelihoods)

    def _uptade_theta_call_likelihood(self, theta: np.array, initial: np.array,
                                      time: np.array, final: np.array,
                                      n_ite: list) -> float:
        """
        Update theta and call the likehood function.

        Since the `optimize` function can only call one function, this
        function is called. It's uptade the theta and call the likelihood.

        Parameters
        ----------
        theta : np.array
            DESCRIPTION.
        initial : np.array
            Initial condition index.
        time : np.array
            Time from one condition index to another.
        final : np.array
            Final condition index.
        n_ite : list
            Current iteration step.

        Returns
        -------
        float
            The likelihood of the model.

        """
        theta[-1] = 0
        n_ite[0] += 1
        self.theta = theta
        likeli = self.likelihood(initial, time, final)
        if self.verbose:
            print(f'likelihood  {n_ite[0]}     = {likeli}',)
        return likeli

    def _optimize_theta(self, initial, time, final):
        """
        Find the optimal theta, which gives the lowest log-likehood

        Parameters
        ----------
        initial : np.array
            Initial condition index.
        time : np.array
            Time from one condition index to another.
        final : np.array
            Final condition index.

        Returns
        -------
        None.

        """
        n_ite = [0]
        bounds = [(0, None) for n in range(self._number_of_states)]
        minimize(
            fun=self._uptade_theta_call_likelihood,
            x0=self.theta,
            args=(initial, time, final, n_ite,),
            method='SLSQP',  # why was this specific algorithm selected?
            bounds=bounds,
            options={'disp': self.verbose}
        )
        self._is_fitted = True

    def _initial_guess_of_theta(self, initial, time, final):
        """
        Initialize theta based on the number and time of transitions.

        Parameters
        ----------
        initial : np.array
            Initial IC.
        time : np.array
            Time between ICs.
        final : np.array
            Final IC.

        Returns
        -------
        None.

        """
        n_transitions = self._number_transitions(initial, final)
        t_transitions = self._time_transitions(initial, time, final)
        self.theta = n_transitions/t_transitions
        # self.theta = np.where(np.isnan(self.theta),0.0000001,self.theta)
        self.theta = np.where(np.isnan(self.theta), 30, self.theta)
        self._is_fitted = True

    def fit(self, initial: np.array, time: np.array, final: np.array):
        """
        Fit the model.

        It initializes the theta based on the number of transitions and the
        time between the traditions. If `self.optimizer` is set as True, the
        theta is optimised until it reaches the most optimal one.

        Parameters
        ----------
        initial : np.array
            Initial IC.
        time : np.array
            Time between ICs.
        final : np.array
            Final IC.

        Returns
        -------
        None.

        """
        # In case the paramters are passed as `list`.
        initial, time, final = np.array(
            initial), np.array(time), np.array(final)
        self._initial_guess_of_theta(initial, time, final)
        if not(self._is_transition_crescent):
            self.theta = np.flip(self.theta)
        if self.verbose:
            print('prior likelihood = ', self.likelihood(initial, time, final))
        if self.optimizer:
            self._optimize_theta(initial, time, final)
        if self.verbose:
            print('posterior likelihood = ',
                  self.likelihood(initial, time, final))

    def get_mean_over_time(self, delta_time: int, initial_IC=None) -> np.array:
        """Get expected IC over time (Analytically)."""
        if initial_IC is None:
            initial_IC = self.best_IC
        initial_IC_index = abs(initial_IC-self.best_IC)  # array position
        mean_IC = np.empty([delta_time + 1])

        # list of possible IC's
        ICs = np.linspace(self.best_IC,
                          self.worst_IC,
                          self._number_of_states,
                          dtype=int)

        for t in range(0, delta_time + 1):
            prob = self.transition_matrix_over_time(t)[initial_IC_index]
            mean_IC[t] = np.array(prob).dot(ICs)
        return mean_IC

    def get_std_over_time(self, delta_time: int, initial_IC=None) -> np.array:
        """Get standard deviation of IC over time (Analytically)."""
        if initial_IC is None:
            initial_IC = self.best_IC
        initial_IC_index = abs(initial_IC-self.best_IC)  # array position
        std_IC = np.empty([delta_time + 1])

        # list of possible IC's
        ICs = np.linspace(self.best_IC,
                          self.worst_IC,
                          self._number_of_states,
                          dtype=int)

        for t in range(0, delta_time + 1):
            prob = self.transition_matrix_over_time(t)[initial_IC_index]
            mean = np.array(prob).dot(ICs)
            var = np.array(prob).dot((ICs - mean)**2)
            std_IC[t] = var**0.5
        return std_IC

    def get_next_IC(self, current_IC: int) -> int:
        """
        Get the next Index Condition (IC).

        Parameters
        ----------
        current_IC : int
            Current IC.

        Returns
        -------
        int
            Next IC.

        """
        IC_index = int(abs(current_IC-self.best_IC))
        prob = self.transition_matrix[IC_index]
        return choices(population=self.list_of_possible_ICs,
                       weights=prob,
                       k=1)[0]

    def predict_MC(self, delta_time: int, initial_IC: int) -> np.array:
        """
        Predict the asset throught time by Monte Carlo approach.

        Parameters
        ----------
        delta_time : int
            DESCRIPTION.
        initial_IC : int
            DESCRIPTION.

        Returns
        -------
        sample : TYPE
            DESCRIPTION.

        """
        #sample = [self.get_next_IC(sample[t-1] for t in range(delta_time))]
        sample = np.empty(delta_time, dtype=int)
        sample[0] = initial_IC
        for t in range(1, delta_time):
            sample[t] = self.get_next_IC(sample[t-1])
        return sample

    def get_mean_over_time_MC(self, delta_time: int,
                              initial_IC: int = None,
                              number_of_samples: int = 1000) -> np.array:
        """
        Get the mean prediction by Monte Carlo approach.

        Parameters
        ----------
        delta_time : int
            DESCRIPTION.
        initial_IC : int, optional
            DESCRIPTION. The default is None.
        number_of_samples : int, optional
            DESCRIPTION. The default is 10000.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if initial_IC is None:
            initial_IC = self.best_IC

        with Pool() as p:
            pool_results = [p.apply_async(self.predict_MC,
                                          (delta_time+1, initial_IC))
                            for _ in range(number_of_samples)]
            samples = [result.get() for result in pool_results]

        return np.mean(samples,
                       axis=0)  # Mean pear year


if __name__ == "__main__":
    import time
    markov = MarkovContinous(worst_IC=5,
                             best_IC=1,
                             optimizer=True,
                             verbose=True)

    initial = np.array([1, 1, 2, 3, 4, 5]*1000)
    t = np.array([1, 2, 3, 4, 5, 5]*1000)
    final = np.array([1, 2, 3, 4, 5, 5]*1000)

    start = time.time()
    markov.fit(initial, t, final)
    print('Time to fit = ',
          round(time.time()-start, 5),
          's. The best so far is = 0.61207 s')

    start = time.time()
    markov.get_mean_over_time(delta_time=100)
    print('Time to get mean (Analytically) = ',
          round(1000*(time.time()-start), 5),
          'ms. The best so far is = 0.14114 ms')

    start = time.time()
    markov.get_mean_over_time_MC(delta_time=100,
                                 number_of_samples=1000)
    print('Time to get mean (Monte Carlo) = ',
          round(time.time()-start, 5),
          's. The best so far is = 1.27329 s')

    # final_IC = markov.get_mean_over_time(delta_time=10, initial_IC=2)[-1]
    # final_IC_MC = markov.get_mean_over_time_MC(delta_time=10, initial_IC=2,
    #                                            number_of_samples=10000)[-1]
    # if 4.407041166528558 != final_IC:
    #     print('ERROR!!!!!!!!!!!')
    #     print('The calculated mean is different than the expected one.')
    #     print('Expected   = ', 4.407041166528558)
    #     print('Calculated = ', final_IC)
    # if abs(final_IC - final_IC_MC) > 0.015:
    #     print('WARNING!!!!!!!!!!!')
    #     print('The error between the calculated mean, Analytically and \
    #           Monte Carlo, is different than the expected one.')
    #     print('Expected = ', 0.015)
    #     print('Calculated = ', abs(final_IC - final_IC_MC))

    # markov = MarkovContinous(worst_IC=1,
    #                          best_IC=5,
    #                          optimizer=True)

    # initial = np.array([1, 1, 2, 3, 4, 5]*1000)
    # t = np.array([1, 2, 3, 4, 5, 5]*1000)
    # final = np.array([1, 2, 3, 4, 5, 5]*1000)
    # markov.fit(initial, t, final)
    # markov.fit(initial, t, final)
