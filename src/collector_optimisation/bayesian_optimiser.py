#!/usr/bin/python3.10
########################################################################################
# bayesian_optimiser.py - Module responsible for carrying out Bayesian optimisations.  #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2024                                                      #
########################################################################################

"""
The optimisation module for the collector-optimisation software.

As part of the optimisation process, this module wraps around external optimisation
libraries in order to expose simple APIs capable of carrying out the optimisations which
take place.

"""

from bayes_opt import BayesianOptimization

from .model_wrapper import PVTModelAssessor

class BayesianPVTModelOptimiser(BayesianOptimization):
    """
    Runs a Bayesian optimisation the PVTModel.

    """

    pass