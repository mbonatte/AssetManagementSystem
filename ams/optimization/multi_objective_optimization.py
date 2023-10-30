"""
Created on Sep 21, 2022.

@author: MauricioBonatte
@e-mail: mbonatte@ymail.com
"""

import json
import numpy as np
from pathlib import Path
THIS_FOLDER = Path(__file__).parent.resolve()

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
    def __init__(self, config={}):
        """
    
        config = {
            "algorithm": 
                {
                "name": "NSGA2",
                "pop_size": 50,
                "eliminate_duplicates": true
                },
            "termination": 
                {
                "name": "n_gen",
                "n_max_gen": 50
                }
        }
        
        """
        self.problem = None
        self.algorithm = None
        self.termination = None
        self.save_history = True
        self.verbose = True

        self._set_algorithm(config.get("algorithm"))
        self._set_termination(config.get("termination"))

    def minimize(self):
        result = minimize(self.problem,
                          self.algorithm,
                          self.termination,
                          save_history = self.save_history,
                          return_least_infeasible = True,
                          verbose = self.verbose)
        return result
        
    def _set_algorithm(self, algorithm):
        if algorithm is None:
            algorithm = {'name': 'NSGA2', 'pop_size': 10, 'eliminate_duplicates': True}
    
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

    def _set_termination(self, termination):
        if termination is None:
            termination = {'name': 'n_gen', 'n_max_gen': 4}
        
        if termination['name'] == 'n_gen':
            termination = get_termination("n_gen", termination['n_max_gen'])
        
        self.termination = termination

    def set_problem(self, problem):
        self.problem = problem