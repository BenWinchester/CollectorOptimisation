#!/usr/bin/python3.10
########################################################################################
# model_wrapper.py - Module to wrap around the individual collector models.            #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2024                                                      #
########################################################################################

"""
The model-wrapper module for the collector-optimisation software.

As part of the optimisation process, this module wraps around the collector models in
order to expose a simple function which can be optimised for fitness based on the
results of the more complex models underneath.

"""

import abc

from typing import Any

class CollectorModelAssessor(abc.ABC):
    """
    Contains attributes and methods to run and optimise the collector model.

    .. attribute:: fitness_function
        A function used for calculating the fitness.

    .. attribute:: weighting_function
        A function used for adjusting the weighting of various outputs.

    """

    def __init__(self, weighting_function: callable) -> None:
        """
        Instnatiate a collector-model assessor.

        :param: weighting_function
            The weighting function to use.

        """

        self.weighting_function: callable = weighting_function

    @abc.abstractmethod
    def fitness_function(self) -> float:
        """
        Calculate and determine the fitness of a run which has taken place.

        :return: The fitness of the run as a `float`.

        """

        pass

class PVTModelAssessor(CollectorModelAssessor):
    """
    Class used for assessing the BenWinchester/PVTModel code.

    """

    def _run_model(self) -> Any:
        """
        Run the model.

        :returns:
            The results of the model.

        """

    def fitness_function(self) -> float:
        """
        Fitness function to assess the fitness of the model.

        :returns: The fitness of the model.

        """

        # Make temporary files as needed based on the inputs for the run.

        # Run the model.

        # Assess the fitness of the results.

        # Return these, deleting the temporary file(s) on exit.
