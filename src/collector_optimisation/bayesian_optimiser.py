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

import functools
import threading

from bayes_opt import BayesianOptimization

from .model_wrapper import PVTModelAssessor

__all__ = (
    "BayesianPVTModelOptimiserSeries",
    "BayesianPVTModelOptimiserThread",
)


class BayesianPVTModelOptimiserSeries:
    """
    Runs a Bayesian optimisation of the PVTModel as a series computation.

    .. attribute:: bayestian_optimiser
        A Bayesian optimiser.

    .. attribute:: optimisation_parameters
        The parameters to pass to the Bayesian optimiser.

    .. attribute:: pvt_model_assessor
        The pvt model assessor.

    .. attribute:: run_id
        The ID associated with the run.

    """

    def __init__(
        self,
        optimisation_parameters: dict[str, tuple[float, float]],
        pvt_model_assessor: PVTModelAssessor,
        run_id_to_results_map: dict[int, dict[str, dict[str, float] | float]],
        solar_irradiance_data: list[float],
        temperature_data: list[float],
        wind_speed_data: list[float],
        *,
        run_id: int,
        random_state: int = 1,
    ) -> None:
        """
        Instantiate the thread.

        :param: optimisation_parameters
            The optimisation parameters used for specifying the bounds of the optimisation.

        :param: pvt_model_assesssor
            The PVT model assessor.

        :param: run_id
            A unique ID for the run, usually just the index of the thread in the number
            of threads that were called.

        :param: random_state
            A random state to use for the Bayesian optimisation.

        """

        self.optimisation_parameters = optimisation_parameters
        self.pvt_model_assessor = pvt_model_assessor
        self.run_id = run_id
        self.run_id_to_results_map = run_id_to_results_map
        self.bayesian_optimiser = BayesianOptimization(
            f=functools.partial(
                pvt_model_assessor.fitness_function,
                run_number=run_id,
                solar_irradiance_data=solar_irradiance_data,
                temperature_data=temperature_data,
                wind_speed_data=wind_speed_data,
            ),
            pbounds=optimisation_parameters,
            random_state=random_state,
        )

    def run(self) -> dict[str, dict[str, float] | float]:
        """
        Run the thread to compute a value.

        :return:
            The optimised values from the run.

        """

        # Run the optimiser.
        self.bayesian_optimiser.maximize(
            init_points=len(self.optimisation_parameters), n_iter=5
        )

        # Save the result and return.
        self.run_id_to_results_map[self.run_id] = (
            optimum_values := self.bayesian_optimiser.max
        )
        return optimum_values


class BayesianPVTModelOptimiserThread(threading.Thread):
    """
    Runs a Bayesian optimisation the PVTModel as a stand-alone thread.

    .. attribute:: bayestian_optimiser
        A Bayesian optimiser.

    .. attribute:: optimisation_parameters
        The parameters to pass to the Bayesian optimiser.

    .. attribute:: pvt_model_assessor
        The pvt model assessor.

    .. attribute:: run_id
        The ID associated with the run.

    """

    def __init__(
        self,
        optimisation_parameters: dict[str, tuple[float, float]],
        pvt_model_assessor: PVTModelAssessor,
        run_id_to_results_map: dict[str, dict[str, float] | float],
        solar_irradiance_data: list[float],
        temperature_data: list[float],
        wind_speed_data: list[float],
        *,
        run_id: int,
        random_state: int = 1,
    ) -> None:
        """
        Instantiate the thread.

        :param: optimisation_parameters
            The optimisation parameters used for specifying the bounds of the optimisation.

        :param: pvt_model_assesssor
            The PVT model assessor.

        :param: run_id
            A unique ID for the run, usually just the index of the thread in the number
            of threads that were called.

        :param: random_state
            A random state to use for the Bayesian optimisation.

        """

        self.optimisation_parameters = optimisation_parameters
        self.pvt_model_assessor = pvt_model_assessor
        self.run_id = run_id
        self.run_id_to_results_map = run_id_to_results_map
        self.bayesian_optimiser = BayesianOptimization(
            f=functools.partial(
                pvt_model_assessor.fitness_function,
                run_number=run_id,
                solar_irradiance_data=solar_irradiance_data,
                temperature_data=temperature_data,
                wind_speed_data=wind_speed_data,
            ),
            pbounds=optimisation_parameters,
            random_state=random_state,
        )

        super().__init__()

    def run(self) -> None:
        """
        Run the thread to compute a value.

        """

        self.bayesian_optimiser.maximize(
            init_points=len(self.optimisation_parameters), n_iter=5
        )

        self.run_id_to_result_map[self.run_id] = self.bayesian_optimiser.max
