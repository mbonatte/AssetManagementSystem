"""
Created on Sep 21, 2022.

@author: MauricioBonatte
@e-mail: mbonatte@ymail.com
"""

import json
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling

from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair

from pymoo.termination import get_termination

from pymoo.optimize import minimize


class Multi_objective_optimization():
    def __init__(self):
        self.problem = None
        self.algorithm = None
        self.termination = None
        self.seed = 1
        self.save_history = True
        self.verbose = True

        self._read_config_file()

    def minimize(self):
        result = minimize(self.problem,
                          self.algorithm,
                          self.termination,
                          # seed=self.seed,
                          save_history=self.save_history,
                          verbose=self.verbose)
        return result

    def set_problem(self, problem):
        self.problem = problem

    def _read_config_file(self):
        file = 'database/optimization_config.json'
        with open(file, 'r') as f:
            data = json.load(f)
        self._set_algorithm(data.get('algorithm', None))
        self._set_termination(data.get('termination', None))

    def _set_algorithm(self, algorithm=None):
        if algorithm['name'] == 'NSGA2':
            algorithm = NSGA2(pop_size=algorithm['pop_size'],
                              sampling=IntegerRandomSampling(),
                              crossover=SBX(prob=1.0,
                                            eta=3.0,
                                            vtype=int,
                                            repair=RoundingRepair()),
                              mutation=PM(prob=1.0,
                                          eta=3.0,
                                          vtype=int,
                                          repair=RoundingRepair()),
                              eliminate_duplicates=algorithm['eliminate_duplicates'],
                              save_history=True)

        self.algorithm = algorithm

    def _set_termination(self, termination=None):
        if termination is None:
            termination = get_termination("n_gen", 4)
        elif termination['name'] == 'n_gen':
            termination = get_termination("n_gen", termination['n_max_gen'])
        self.termination = termination
