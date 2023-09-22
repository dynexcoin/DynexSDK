import dynex
from dimod.binary_quadratic_model import BinaryQuadraticModel
import numpy as np


class Sampler(object):
    """
    This module defines a sampler.
    :param num_samps: number of samples
    :type num_samps: int
    """

    def __init__(self, num_copies=1):
        self.solver = 'DYNEX';

    def sample_qubo(self, Q, num_samps=100):
        """
        Sample from the QUBO problem
        :param qubo: QUBO problem
        :type qubo: numpy dictionary
        :return: samples, energy, num_occurrences
        """
        self.num_samps = num_samps

        if not hasattr(self, 'sampler'):

            bqm = BinaryQuadraticModel.from_qubo(Q)
            # use the dynex sampler:
            dnxmodel = dynex.BQM(bqm, logging=False);
            dnxsampler = dynex.DynexSampler(dnxmodel, logging=False, mainnet=False);
            solution1 = dnxsampler.sample(num_reads=2048, annealing_time = 100, debugging=False);
            
            # TODO: parse multiple samples

            ret1 = []; ret2 = []; ret3 = [];
            for i in range(0,1):
            	ret1.append(np.array(list(solution1.first.sample.values())));
            	ret2.append(solution1.first.energy);
            	ret3.append(1);
            
        return ret1,ret2,ret3
        
        
