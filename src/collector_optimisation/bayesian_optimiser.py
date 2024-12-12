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
import os
import threading

import pandas as pd

from bayes_opt import BayesianOptimization

from .__utils__ import DateAndTime
from .model_wrapper import PVTModelAssessor

__all__ = (
    "BayesianPVTModelOptimiserSeries",
    "BayesianPVTModelOptimiserThread",
    "MAX_RESULTS_FILENAME",
)


# MAX_RESULTS_FILENAME:
#   Filename used for storing the maximum results once the optimisation has taken place.
MAX_RESULTS_FILENAME: str = "max_runs_data_{date}_{time}.csv"

# OPTIMUM_FILE_LOCK:
#   Lock used to lock the file for storing information based on runs and fitness
# information.
OPTIMUM_FILE_LOCK: threading.Lock = threading.Lock()


def _save_max_results(*, date_and_time: DateAndTime, **kwargs) -> None:
    """
    Save information about the current run.

    :param: args
        Arguments to be saved.

    :param: kwargs
        Keyword arguments to be saved.

    """

    try:
        # Acquire the lock on saving the file.
        OPTIMUM_FILE_LOCK.acquire()

        row = pd.DataFrame(
            {
                "target": kwargs["target"],
                "run_number": kwargs["run_number"],
                **{key: [value] for key, value in kwargs["params"].items()},
            }
        )

        # Read any existing runs that have taken place.
        if os.path.isfile(
            MAX_RESULTS_FILENAME.format(
                date=date_and_time.date, time=date_and_time.time
            )
        ):
            with open(
                MAX_RESULTS_FILENAME.format(
                    date=date_and_time.date, time=date_and_time.time
                ),
                "r",
                encoding="UTF-8",
            ) as runs_file:
                runs_data: pd.DataFrame | None = pd.read_csv(runs_file)

        else:
            runs_data = None

        # Append the current run information.
        runs_data = pd.concat([runs_data, row])

        # Write the data to the file
        with open(
            MAX_RESULTS_FILENAME.format(
                date=date_and_time.date, time=date_and_time.time
            ),
            "w",
            encoding="UTF-8",
        ) as runs_file:
            runs_data.to_csv(runs_file, index=None)

    # Release the lock at the end of attempting to save information.
    finally:
        OPTIMUM_FILE_LOCK.release()


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
        date_and_time: DateAndTime,
        optimisation_parameters: dict[str, tuple[float, float]],
        pvt_model_assessor: PVTModelAssessor,
        run_id_to_results_map: dict[int, dict[str, dict[str, float] | float]],
        solar_irradiance_data: list[float],
        temperature_data: list[float],
        wind_speed_data: list[float],
        *,
        initial_points: int | None = None,
        num_iterations: int = 5,
        run_id: int,
        random_state: int = 1,
    ) -> None:
        """
        Instantiate the thread.

        :param: date_and_time
            The date and time information to use when saving model results.

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

        self.date_and_time = date_and_time
        self.num_iterations = num_iterations
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

        # Set the number of initial points
        if initial_points is None:
            initial_points = len(self.optimisation_parameters)

        self.num_initial_points = initial_points

    def run(self) -> dict[str, dict[str, float] | float]:
        """
        Run the thread to compute a value.

        :return:
            The optimised values from the run.

        """

        # Run the optimiser.
        self.bayesian_optimiser.maximize(
            init_points=self.num_initial_points, n_iter=self.num_iterations
        )

        # Save the maximum results to a file.
        _save_max_results(
            date_and_time=self.date_and_time,
            run_number=self.run_id,
            **self.bayesian_optimiser.max,
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
        date_and_time: DateAndTime,
        optimisation_parameters: dict[str, tuple[float, float]],
        pvt_model_assessor: PVTModelAssessor,
        run_id_to_results_map: dict[str, dict[str, float] | float],
        solar_irradiance_data: list[float],
        temperature_data: list[float],
        wind_speed_data: list[float],
        *,
        initial_points: int | None = None,
        num_iterations: int = 5,
        run_id: int,
        random_state: int = 1,
    ) -> None:
        """
        Instantiate the thread.

        :param: date_and_time
            The date and time information to use when saving model results.

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

        self.date_and_time = date_and_time
        self.num_iterations = num_iterations
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

        # Set the number of initial points
        if initial_points is None:
            initial_points = len(self.optimisation_parameters)

        self.num_initial_points = initial_points

        super().__init__()

    def run(self) -> None:
        """
        Run the thread to compute a value.

        """

        self.bayesian_optimiser.maximize(
            init_points=self.num_initial_points, n_iter=self.num_iterations
        )

        # Save the maximum results to a file.
        _save_max_results(
            date_and_time=self.date_and_time,
            run_number=self.run_id,
            **self.bayesian_optimiser.max,
        )

        self.run_id_to_results_map[self.run_id] = self.bayesian_optimiser.max
