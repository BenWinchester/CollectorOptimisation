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
import enum
import os
import random
import sys
import threading

from contextlib import contextmanager
from io import StringIO
from typing import Any, Generator, Type, TypeVar

import numpy as np
import pandas as pd
import yaml

from pvt_model import main as pvt_model_main
from pvt_model import SystemData

from .__utils__ import DateAndTime, INPUT_FILES_DIRECTORY, WeatherDataHeader

# FILE_LOCK:
#   Lock used to lock the file for storing information based on runs and fitness
# information.
FILE_LOCK: threading.Lock = threading.Lock()

# LOCATIONS_FOLDERNAME:
#   The name of the folder where locations are stored.
LOCATIONS_FOLDERNAME: str = "locations"

# MAX_PARALLEL_RUNS:
#   The maximum number of possible parallel runs, used for id's in the case that these
#   aren't provided to context managers.
MAX_PARALLEL_RUNS: int = 10000

# RUNS_DATA_FILENAME:
#   The name of the runs data file.
RUNS_DATA_FILENAME: str = "runs_data_{date}_{time}.csv"

# TEMPORARY_FILE_DIRECTORY:
#   The name of the temporary file directory to use.
TEMPORARY_FILE_DIRECTORY: str = "temp"


# Type variable for capturing contet manager.
C = TypeVar(
    "C",
    bound="Capturing",
)


class Capturing(list):
    """
    Context manager for capturing calls to stdout.

    This class comes from Kindal on StackExchange:
    https://stackoverflow.com/a/16571630

    """

    def __enter__(self) -> Type[C]:
        """
        Enter the context manager.

        Sets up a prive variable where the stdout calls is stored and returns this
        instance.

        """

        self._stdout = sys.stdout
        # self._stderr = sys.stderr
        sys.stdout = self._stringioout = StringIO()
        # sys.stderr = self._stringioerr = StringIO()
        return self

    def __exit__(self, *args) -> None:
        """
        Exit the context manager.

        Stores the current value of the stdout call and returns, then deletes to free up
        memory once the `list` has been returned.

        NOTE: Because this class inherits from `list`, it can return itself as a list.

        """

        self.extend(self._stringioout.getvalue().splitlines())
        # self.extend(self._stringioerr.getvalue().splitlines())
        del self._stringioout  # free up some memory
        # del self._stringioerr  # free up some memory
        sys.stdout = self._stdout
        # sys.stderr = self._stderr


def _save_current_run(*, date_and_time: DateAndTime, **kwargs) -> None:
    """
    Save information about the current run.

    :param: args
        Arguments to be saved.

    :param: kwargs
        Keyword arguments to be saved.

    """

    # Acquire the lock on saving the file.
    FILE_LOCK.acquire()

    row = pd.DataFrame({key: [value] for key, value in kwargs.items()})

    try:
        # Read any existing runs that have taken place.
        if os.path.isfile(
            RUNS_DATA_FILENAME.format(date=date_and_time.date, time=date_and_time.time)
        ):
            with open(
                RUNS_DATA_FILENAME.format(
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
            RUNS_DATA_FILENAME.format(date=date_and_time.date, time=date_and_time.time),
            "w",
            encoding="UTF-8",
        ) as runs_file:
            runs_data.to_csv(runs_file, index=None)

    # Release the lock at the end of attempting to save information.
    finally:
        FILE_LOCK.release()


@contextmanager
def temporary_collector_file(
    base_collector_filepath: str,
    date_and_time: DateAndTime,
    updates_to_collector_design_parameters: dict[str, float],
    unique_id: int = random.randint(0, MAX_PARALLEL_RUNS),
) -> Generator[str, None, None]:
    """
    Create, manage, and delete a temporary collector file.

    :param: base_collector_filepath
        The path to the base collector file.

    :param: updates_to_collector_design_parameters
        A mapping between named design parameters and values to update them with.

    :param: unique_id
        A unique ID for the run.

    :yields: The path to the temporary file.

    """

    def _vary_parameter(data: dict[str, Any], key: str, value: float):
        """
        Vary the parameter and return it.

        :param: data
            The data to update.

        :param: key
            The name of the parameter, which should utilise forward slashes between
            names.

        :param: value
            The value to update the parameter with.

        """

        current_key: str = key.split("/")[0]
        next_key = "/".join(key.split("/")[1:])

        if next_key == "":
            # Make the substitution and return if at the bottom.
            data[current_key] = float(value)

            # If the value is absorptivity, reduce the transmissivity respectively.
            if current_key == "absorptivity":
                data["transmissivity"] = float(1 - value)
            if current_key == "transmissivity":
                data["absorptivity"] = float(1 - value)

            return

        # Otherwise, if another stage is needed, call the self using a subset of the
        # data.
        return _vary_parameter(data[current_key], next_key, value)

    # Open the data and parse the data.
    with open(base_collector_filepath, "r", encoding="UTF-8") as collector_file:
        base_collector_data = yaml.safe_load(collector_file)

    # Attempt to loop through and update with all the parameters.
    for key, value in updates_to_collector_design_parameters.items():
        _vary_parameter(base_collector_data, key, value)

    # Make the temporary directory if it doesn't exist.
    if not os.path.isdir(TEMPORARY_FILE_DIRECTORY):
        os.makedirs(TEMPORARY_FILE_DIRECTORY, exist_ok=True)

    # Save these data to a temporary file
    try:
        with open(
            (
                filename := os.path.join(
                    TEMPORARY_FILE_DIRECTORY,
                    f"{os.path.basename(base_collector_filepath).split('.')[0]}_"
                    f"{unique_id}_{date_and_time.date}_{date_and_time.time}.yaml",
                )
            ),
            "w",
            encoding="UTF-8",
        ) as temp_file:
            yaml.dump(base_collector_data, temp_file)

        yield filename

    finally:
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass


@contextmanager
def temporary_steady_state_file(
    base_steady_state_filepath: str,
    date_and_time: DateAndTime,
    mass_flow_rate: float,
    solar_irradiance_data: list[float],
    temperature_data: list[float],
    wind_speed_data: list[float],
    unique_id: int = random.randint(0, MAX_PARALLEL_RUNS),
) -> Generator[str, None, None]:
    """
    Create and return the path to a temporary steady-state file.

    :param: base_steady_state_filepath
        The path to the steady-state file on which to base the run.

    :param: mass_flow_rate
        The mass flow rate to use for this run.

    :param: solar_irradiance_data
        The solar-irradiance data to use for the run.

    :param: temperature_data
        The temperature data for the run.

    :param: wind_speed_data
        The wind-speed data for the run.

    :param: unique_id
        A unique ID for the run.

    """

    with open(base_steady_state_filepath, "r", encoding="UTF-8") as steady_state_file:
        base_steady_state_data = yaml.safe_load(steady_state_file)

    # Assert that all input data is of the same length.
    assert len(solar_irradiance_data) == len(temperature_data) == len(wind_speed_data)

    # Generate a dataframe to contain the information.
    data_frame = pd.DataFrame(
        {
            WeatherDataHeader.SOLAR_IRRADIANCE.value: solar_irradiance_data.repeat(
                len(base_steady_state_data)
            ),
            WeatherDataHeader.AMBIENT_TEMPERATURE.value: temperature_data.repeat(
                len(base_steady_state_data)
            ),
            WeatherDataHeader.WIND_SPEED.value: wind_speed_data.repeat(
                len(base_steady_state_data)
            ),
            "mass_flow_rate": [mass_flow_rate]
            * len(wind_speed_data)
            * len(base_steady_state_data),
            "collector_input_temperature": [
                entry["collector_input_temperature"] for entry in base_steady_state_data
            ]
            * len(wind_speed_data),
        }
    )

    # Save these data to a temporary file
    try:
        with open(
            (
                filename := os.path.join(
                    TEMPORARY_FILE_DIRECTORY,
                    f"{os.path.basename(base_steady_state_filepath).split('.')[0]}_"
                    f"{unique_id}_{date_and_time.date}_{date_and_time.time}.csv",
                )
            ),
            "w",
            encoding="UTF-8",
        ) as temp_file:
            data_frame.to_csv(temp_file, index=None)

        yield filename

    finally:
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass


class Fitness(float):
    """
    Used to represent fitness whilst containing additional attributes.

    .. attribute:: electrical_fitness
        The electrical fitness value.

    .. attribute:: thermal_fitness
        The thermal fitness value.

    """

    def __new__(
        cls, combined_fitness: float, electrical_fitness: float, thermal_fitness: float
    ):
        """Override the __new__ method to provide features."""

        _instance = super().__new__(cls, combined_fitness)
        _instance.electrical_fitness = electrical_fitness
        _instance.thermal_fitness = thermal_fitness

        return _instance


class WeightingCalculator:
    """
    Contains functionality for calculating the weighting between outputs.

    """

    def __init__(self, electrical_weighting: float, thermal_weighting: float) -> None:
        """
        Instantiate the weighting-calculator instance.

        :param: electrical_weighting
            The weighting to give to the electrical output.

        :param: thermal_weighting
            The weighting to give to the thermal output.

        """

        self.electrical_weighting = electrical_weighting
        self.thermal_weighting = thermal_weighting
        self.total_output_weighting = electrical_weighting + thermal_weighting

    def __repr__(self) -> str:
        """Return a default representation of the class."""

        return (
            f"WeightingCalculator(el={self.electrical_weighting:.2g}, th="
            + f"{self.thermal_weighting:.2g})"
        )

    @property
    def name(self) -> str:
        """
        Return a name used for identifying and saving information.

        """

        return f"{self.electrical_weighting:.2g}_el_{self.thermal_weighting:.2g}_th"

    def get_weighted_fitness(
        self, electrical_fitness: float, thermal_fitness: float
    ) -> Fitness:
        """
        Calculate and return a combined fitness based on electrical and thermal values.

        :param: electrical_fitness
            The current value of the electrical output/fitness.

        :param: thermal_fitness
            The current value of the thermal output/fitness.

        :returns: The weighted fitness.

        """

        return Fitness(
            (self.electrical_weighting / self.total_output_weighting)
            * electrical_fitness
            + (self.thermal_weighting / self.total_output_weighting) * thermal_fitness,
            electrical_fitness,
            thermal_fitness,
        )


class CollectorType(enum.Enum):
    """
    Denotes the type of collector being optimised.

    - PVT:
        A PV-T collector.

    """

    PVT = "pvt"


# Type variable for collector model assessor and children.
CMA = TypeVar(
    "CMA",
    bound="CollectorModelAssessor",
)


class CollectorModelAssessor(abc.ABC):
    """
    Contains attributes and methods to run and optimise the collector model.

    .. attribute:: fitness_function
        A function used for calculating the fitness.

    .. attribute:: weighting_function
        A function used for adjusting the weighting of various outputs.

    """

    collector_type: CollectorType
    collector_type_to_wrapper: dict[CollectorType, CMA] = {}

    def __init__(self, weighting_calculator: WeightingCalculator) -> None:
        """
        Instnatiate a collector-model assessor.

        :param: weighting_function
            The weighting function to use.

        """

        self.weighting_calculator: WeightingCalculator = weighting_calculator

    def __init_subclass__(cls: Type[CMA], collector_type: CollectorType) -> None:
        cls.collector_type = collector_type
        cls.collector_type_to_wrapper[collector_type] = cls
        return super().__init_subclass__()

    @abc.abstractmethod
    def fitness_function(self) -> float:
        """
        Calculate and determine the fitness of a run which has taken place.

        :return: The fitness of the run as a `float`.

        """

        pass


class PVTModelAssessor(CollectorModelAssessor, collector_type=CollectorType.PVT):
    """
    Class used for assessing the BenWinchester/PVTModel code.

    """

    def __init__(
        self,
        base_pvt_filepath: str,
        base_model_input_files: list[str],
        date_and_time: DateAndTime,
        location_name: str,
        output_filename: str,
        *,
        weighting_calculator: WeightingCalculator,
    ) -> None:
        """
        Instnatiate the class.

        :param: base_pvt_filepath
            The base PV-T filepath from which changes to the collector will be explored.

        :param: base_steadystate_filepath
            The base steady-state filepath from which new runs will be generated.

        :param: date_and_time
            The date and time to use when saving results.

        :param: location_name
            Deprecated---the location name utilised for the weather data setup, required
            by PVTModel.

        :param: output_filename
            The output filename

        :param: weighting_calculator
            The weighting calculator to use to determine the combined metric for
            performance.

        """

        # Process the model--input file information.
        try:
            base_steadystate_filepath: str = base_model_input_files[0]
        except TypeError:
            raise Exception(
                "Base model-input files need to be specified on the CLI."
            ) from None
        except IndexError:
            raise Exception(
                "Expected one model-input file for the model type, "
                f"{self.collector_type.value}. None were provided."
            )

        self.base_pvt_filepath: str = base_pvt_filepath
        self.base_steady_state_filepath: str = base_steadystate_filepath
        self.date_and_time = date_and_time
        self.model_args: list[str] = [
            "--skip-analysis",
            "--output",
            output_filename,
            "--decoupled",
            "--steady-state",
            "--initial-month",
            "7",
            "--location",
            os.path.join(INPUT_FILES_DIRECTORY, LOCATIONS_FOLDERNAME, location_name),
            "--portion-covered",
            "1",
            "--x-resolution",
            "31",
            "--y-resolution",
            "11",
            "--average-irradiance",
            "--skip-2d-output",
            "--layers",
            "g",
            "pv",
            "a",
            "p",
            "f",
            "--disable-logging",
            "--skip-output",
        ]
        self.output_filename = output_filename

        super().__init__(weighting_calculator)

    def _run_model(
        self, temporary_pvt_filepath: str, temporary_steady_state_filepath: str
    ) -> dict[float, SystemData]:
        """
        Run the model.

        :param: temporary_pvt_filepath
            The temporary filepath to the PV-T file to use.

        :param: temporary_steady_state_filepath
            The temporary filepath to the steady-state file to use.

        :returns:
            The results of the model.

        """

        model_args = self.model_args + [
            "--steady-state-data-file",
            temporary_steady_state_filepath,
        ]
        model_args.extend(["--pvt-data-file", temporary_pvt_filepath])

        # Remove the output file if it already exists.
        if os.path.isfile(self.output_filename):
            os.remove(self.output_filename)

        with Capturing():
            return pvt_model_main(model_args)

    def unweighted_fitness_function(
        self,
        mass_flow_rate: float,
        run_number: int,
        solar_irradiance_data: list[float],
        temperature_data: list[float],
        wind_speed_data: list[float],
        **kwargs,
    ) -> tuple[float, float, dict[float, SystemData]]:
        """
        Calculate the un-weighted, i.e., separate fitness for electricity and heat.

        Operation Parameters:
            :param: mass_flow_rate
                The mass flow rate through the collector.

        :param: run_number
            The run number.

        :param: run_weightings
            The weightings to use for each result of the run.

        :param: solar_irradiance_data
            The solar-irradiance data to use for the run.

        :param: temperature_data
            The temperature data for the run.

        :param: wind_speed_data
            The wind-speed data for the run.

        Design Parameters:
            These are passed in with the kwargs parameters and determined based on this.

        :returns:
            - The electrical fitness of the model,
            - The thermal fitness of the model.

        """

        # Make temporary files as needed based on the inputs for the run.
        with temporary_collector_file(
            self.base_pvt_filepath, self.date_and_time, kwargs, run_number
        ) as temp_pvt_filepath:
            with temporary_steady_state_file(
                self.base_steady_state_filepath,
                self.date_and_time,
                mass_flow_rate,
                solar_irradiance_data,
                temperature_data,
                wind_speed_data,
                run_number,
            ) as temp_steady_state_filepath:
                # Run the model.
                output_data = self._run_model(
                    temp_pvt_filepath, temp_steady_state_filepath
                )

        # Use the run weights for each of the runs that were returned.
        electrical_fitness = np.sum(
            entry.electrical_power for entry in output_data.values()
        )
        thermal_fitness = np.sum(entry.thermal_power for entry in output_data.values())

        # Return these fitnesses.
        return electrical_fitness, thermal_fitness, output_data

    def fitness_function(
        self,
        mass_flow_rate: float,
        run_number: int,
        solar_irradiance_data: list[float],
        temperature_data: list[float],
        wind_speed_data: list[float],
        **kwargs,
    ) -> float:
        """
        Fitness function to assess the fitness of the model.

        Operation Parameters:
            :param: mass_flow_rate
                The mass flow rate through the collector.

        :param: run_number
            The run number.

        :param: run_weightings
            The weightings to use for each result of the run.

        :param: solar_irradiance_data
            The solar-irradiance data to use for the run.

        :param: temperature_data
            The temperature data for the run.

        :param: wind_speed_data
            The wind-speed data for the run.

        Design Parameters:
            These are passed in with the kwargs parameters and determined based on this.

        :returns: The fitness of the model.

        """

        # Calculate the unweighted fitnesses.
        electrical_fitness, thermal_fitness, output_data = (
            self.unweighted_fitness_function(
                mass_flow_rate,
                run_number,
                solar_irradiance_data,
                temperature_data,
                wind_speed_data,
                **kwargs,
            )
        )

        # Assess the fitness of the results and return.
        weighted_fitness = self.weighting_calculator.get_weighted_fitness(
            electrical_fitness, thermal_fitness
        )

        _save_current_run(
            date_and_time=self.date_and_time,
            electrical_fitness=electrical_fitness,
            fitness=weighted_fitness,
            thermal_fitness=thermal_fitness,
            mass_flow_rate=mass_flow_rate,
            run_number=run_number,
            **kwargs,
        )

        if electrical_fitness < 0 or thermal_fitness < 0 or weighted_fitness < 0:
            import pdb

            pdb.set_trace(
                header=(
                    "Negative fitness is "
                    ", ".join(
                        [
                            entry
                            for entry in (
                                electrical_fitness,
                                thermal_fitness,
                                weighted_fitness,
                            )
                            if entry < 0
                        ]
                    )
                )
            )

        return weighted_fitness
