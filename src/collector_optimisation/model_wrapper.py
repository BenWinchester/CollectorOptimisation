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
import itertools
import json
import os
import queue
import random
import shutil
import subprocess
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
from tqdm import tqdm

from .__utils__ import DateAndTime, INPUT_FILES_DIRECTORY, WeatherDataHeader

# FILE_LOCK:
#   Lock used to lock the file for storing information based on runs and fitness
# information.
FILE_LOCK: threading.Lock = threading.Lock()

# HUANG_ET_AL_DIRECTORY
#   Directory name for the Huang _et al._ model.
HUANG_ET_AL_DIRECTORY: str = "huang_et_al_sspvt"

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

# SSPVT_BAYESIAN_OUTPUT_DIRECTORY:
#   Output directory into which SSPV-T outputs from the Bayesian optimisation are saved.
SSPVT_BAYESIAN_OUTPUT_DIRECTORY: str = "sspvt_bayesian_output"

# SSPVT_BAYESIAN_OUTPUT_FILENAME:
#   Ouptut filename for Bayesian output files.
SSPVT_BAYESIAN_OUTPUT_FILENAME: str = "results_run_{suffix}_{panel_filename}.json"

# SSPVT_SUFFIX_COUNTER:
#   A counter used to provide a suffix when running parallel SSPV-T computations
SSPVT_SUFFIX_QUEUE: queue.Queue = queue.Queue()

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


def sspvt_model_main(panel_filename: str, suffix: str) -> None:
    """
    Run the SSPV-T model, written in MATLAB by Gan Huang _et al._.

    :param: panel_filename
        The name of the panel file.

    :param: suffix
        The unique suffix to use for the run.

    """

    import pdb

    pdb.set_trace(header="SSPV-T launcher")

    # Run the model from the SSPVT directory.
    try:
        os.chdir(HUANG_ET_AL_DIRECTORY)
        subprocess.run(
            (
                'matlab -nodesktop -nosplash -nojvm -r "panel_filename='
                + f"'{os.path.basename(panel_filename)}';"
                + f'suffix={suffix};sspvt_bayesian"'
            ).split(" ")
        )

    # Change up a directory.
    finally:
        # Try to move the output file up a directory.
        try:
            shutil.copy2(
                output_filename := SSPVT_BAYESIAN_OUTPUT_FILENAME.format(
                    panel_filename=panel_filename, suffix=suffix
                ),
                os.path.join("..", output_filename),
            )
        except Exception:
            pass

        # Try to return the process up a directory.
        try:
            os.chdir("..")
        except Exception:
            pass


@contextmanager
def temporary_collector_file(
    base_collector_filepath: str,
    date_and_time: DateAndTime,
    updates_to_collector_design_parameters: dict[str, float],
    temp_upper_dirname: str | None = None,
    unique_id: int = random.randint(0, MAX_PARALLEL_RUNS),
) -> Generator[tuple[str, float | None, float | None], None, None]:
    """
    Create, manage, and delete a temporary collector file.

    :param: base_collector_filepath
        The path to the base collector file.

    :param: updates_to_collector_design_parameters
        A mapping between named design parameters and values to update them with.

    :param: unique_id
        A unique ID for the run.

    :yields:
        - The path to the temporary file,
        - the width of the collector used when running the simulation,
        - and the initial width of the collector.

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

        # Recursively update the data if required.
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

    class Loader(enum.Enum):
        YAML = "yaml"
        CSV = "csv"
        JSON = "json"

    # Open the data and parse the data.
    with open(base_collector_filepath, "r", encoding="UTF-8") as collector_file:
        if base_collector_filepath.endswith(".yaml"):
            base_collector_data = yaml.safe_load(collector_file)
            loader: Loader = Loader.YAML

        elif base_collector_filepath.endswith(".csv"):
            base_collector_data = pd.read_csv(collector_file)
            loader = Loader.CSV

        elif base_collector_filepath.endswith(".json"):
            base_collector_data = json.load(collector_file)
            loader = Loader.JSON

        else:
            raise Exception("File not implemented.")

    # Determine the initial width of the collector.
    try:
        initial_collector_width = (
            base_collector_data["pvt_collector"]["width"]
            * base_collector_data["number_of_modelled_segments"]
        )
    except KeyError:
        print("No initial collector width. Skipping.")
        initial_collector_width = None

    # Attempt to loop through and update with all the parameters.
    for key, value in updates_to_collector_design_parameters.items():
        if key in base_collector_data:
            base_collector_data[key] = value
            continue

        _vary_parameter(base_collector_data, key, value)

    # Make the temporary directory if it doesn't exist.
    if temp_upper_dirname is not None:
        os.makedirs(temp_upper_dirname, exist_ok=True)
        temp_dirname: str = os.path.join(temp_upper_dirname, TEMPORARY_FILE_DIRECTORY)
        # temp_dirname: str = temp_upper_dirname

    else:
        temp_dirname = TEMPORARY_FILE_DIRECTORY

    if not os.path.isdir(temp_dirname):
        os.makedirs(temp_dirname, exist_ok=True)

    # Save these data to a temporary file
    try:
        with open(
            (
                filename := os.path.join(
                    temp_dirname,
                    f"temp_{os.path.basename(base_collector_filepath).split('.')[0]}_"
                    f"{unique_id}_{date_and_time.date}_{date_and_time.time}." + loader.value,
                )
            ),
            "w",
            encoding="UTF-8",
        ) as temp_file:
            if loader == Loader.YAML:
                yaml.dump(base_collector_data, temp_file)
            elif loader == Loader.CSV:
                base_collector_data.to_csv(temp_file, header=None)
            elif loader == Loader.JSON:
                json.dump(base_collector_data, temp_file, indent=4)

        try:
            current_collector_width: float | None = base_collector_data[
                "pvt_collector"
            ]["width"]
        except KeyError:
            current_collector_width = None

        yield filename, current_collector_width, initial_collector_width

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
                    f"temp_{unique_id}_{date_and_time.date}_{date_and_time.time}.csv",
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


@contextmanager
def temporary_sspvt_steady_state_files(
    base_ambient_temperature_input_filepath: str,
    base_coolant_input_filepath: str,
    base_fluid_input_filepath: str,
    base_irradiance_input_filepath: str,
    base_wind_speed_input_filepath: str,
    solar_irradiance_data: list[float],
    temperature_data: list[float],
    wind_speed_data: list[float],
    temp_upper_dirname: str | None = None,
) -> Generator[str, None, None]:
    """
    Create and return a suffix for the paths to the temporary steady-state file(s).

    The steady-state files are generated, and a suffix is returned.

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

    # Assert that all input data is of the same length.
    assert len(solar_irradiance_data) == len(temperature_data) == len(wind_speed_data)

    # Determine the length of the input files required.
    with open(base_coolant_input_filepath, "r", encoding="UTF-8") as coolant_file:
        base_coolant_input_data = pd.read_csv(coolant_file, header=None).transpose()

    with open(base_fluid_input_filepath, "r", encoding="UTF-8") as fluid_file:
        base_fluid_input_data = pd.read_csv(fluid_file, header=None).transpose()

    # Assert that these files are of the same length
    assert len(base_coolant_input_data) == len(base_fluid_input_data)

    # Repeat the coolant and fluid input data the number of times required.
    coolant_temperatures = sorted(set(base_coolant_input_data[0]))
    fluid_temperatures = sorted(set(base_fluid_input_data[0]))

    # Produce modelling temperatures, whereby the arrays are repeated in such a way that
    # the weather data simply needs to be itertool chained, or pandas concatenated.
    modelling_coolant_temperatures = pd.DataFrame(
        np.repeat(
            np.repeat(coolant_temperatures, len(fluid_temperatures)),
            len(solar_irradiance_data),
        )
    )
    modelling_fluid_temperatures = pd.DataFrame(
        np.repeat(
            list(
                itertools.chain.from_iterable(
                    [fluid_temperatures for _ in range(len(coolant_temperatures))]
                )
            ),
            len(solar_irradiance_data),
        )
    )

    # Repeat the weather data the number of times required.
    number_of_input_temperature_points = len(coolant_temperatures) * len(
        fluid_temperatures
    )
    modelling_ambient_temperature_data = pd.concat(
        [temperature_data for _ in range(number_of_input_temperature_points)], axis=0
    )
    modelling_solar_irradiance_data = pd.concat(
        [solar_irradiance_data for _ in range(number_of_input_temperature_points)],
        axis=0,
    )
    modelling_wind_speed_data = pd.concat(
        [wind_speed_data for _ in range(number_of_input_temperature_points)], axis=0
    )

    if temp_upper_dirname is not None:
        temporary_file_directory: str = os.path.join(
            temp_upper_dirname, TEMPORARY_FILE_DIRECTORY
        )
        # temporary_file_directory: str = temp_upper_dirname
        os.makedirs(temporary_file_directory, exist_ok=True)
    else:
        temporary_file_directory = TEMPORARY_FILE_DIRECTORY

    # Fetch a unique ID to use for the suffix
    try:
        suffix = SSPVT_SUFFIX_QUEUE.get()

        # Save these data to a temporary file
        try:
            for filepath, data_frame in tqdm(
                (
                    temporary_files_and_data := {
                        base_ambient_temperature_input_filepath: modelling_ambient_temperature_data,
                        base_coolant_input_filepath: modelling_coolant_temperatures,
                        base_fluid_input_filepath: modelling_fluid_temperatures,
                        base_irradiance_input_filepath: modelling_solar_irradiance_data,
                        base_wind_speed_input_filepath: modelling_wind_speed_data,
                    }
                ).items(),
                desc="Creating temporary weather files",
                leave=False,
            ):
                os.makedirs(temporary_file_directory, exist_ok=True)
                with open(
                    os.path.join(
                        temporary_file_directory,
                        "temp_"
                        + os.path.basename(filepath.replace(".csv", f"_{suffix}.csv")),
                    ),
                    "w",
                    encoding="UTF-8",
                ) as temp_file:
                    data_frame.to_csv(temp_file, index=None, header=None)

            yield suffix

        finally:
            try:
                for filepath in temporary_files_and_data.keys():
                    try:
                        os.remove(
                            os.path.join(
                                temporary_file_directory,
                                "temp_"
                                + os.path.basename(
                                    filepath.replace(".csv", f"_{suffix}.csv")
                                ),
                            ).replace(".csv", f"_{suffix}.csv")
                        )
                    except FileNotFoundError:
                        pass
            except NameError:
                pass

    finally:
        SSPVT_SUFFIX_QUEUE.put(suffix)


class Fitness(float):
    """
    Used to represent fitness whilst containing additional attributes.

    .. attribute:: electrical_fitness
        The electrical fitness value.

    .. attribute:: thermal_fitness
        The thermal fitness value.

    .. attribute:: coolant_fitness
        The coolant fitness value.

    """

    def __new__(
        cls,
        combined_fitness: float,
        electrical_fitness: float,
        thermal_fitness: float,
        coolant_fitness: float | None,
    ):
        """Override the __new__ method to provide features."""

        _instance = super().__new__(cls, combined_fitness)
        _instance.electrical_fitness = electrical_fitness
        _instance.thermal_fitness = thermal_fitness
        _instance.coolant_fitness = coolant_fitness

        return _instance


class WeightingCalculator:
    """
    Contains functionality for calculating the weighting between outputs.

    """

    def __init__(
        self,
        electrical_weighting: float,
        thermal_weighting: float,
        coolant_weighting: float | None = None,
    ) -> None:
        """
        Instantiate the weighting-calculator instance.

        :param: electrical_weighting
            The weighting to give to the electrical output.

        :param: thermal_weighting
            The weighting to give to the thermal output.

        :param: coolant_weighting
            If relevant, the weighting given to the coolant output.

        """

        self._coolant_weighting = coolant_weighting
        self.electrical_weighting = electrical_weighting
        self.thermal_weighting = thermal_weighting
        self.total_output_weighting = (
            electrical_weighting
            + thermal_weighting
            + (coolant_weighting if coolant_weighting is not None else 0)
        )

    @property
    def coolant_weighting(self) -> float:
        """The coolant weighting, as a float."""

        if self._coolant_weighting is None:
            return 0
        return self._coolant_weighting

    def __repr__(self) -> str:
        """Return a default representation of the class."""

        return (
            f"WeightingCalculator(el={self.electrical_weighting:.2g}, "
            + f"th{'_HT' * self._coolant_weighting is not None}="
            + f"{self.thermal_weighting:.2g}"
            + (
                f", th_LT={self.coolant_weighting:.2g}"
                * (self._coolant_weighting is not None)
            )
            + ")"
        )

    @property
    def name(self) -> str:
        """
        Return a name used for identifying and saving information.

        """

        return f"{self.electrical_weighting:.2g}_el_{self.thermal_weighting:.2g}_th" + (
            f"HT_{self.coolant_weighting:.2g}_th_LT"
            * (self._coolant_weighting is not None)
        )

    def get_weighted_fitness(
        self,
        electrical_fitness: float,
        thermal_fitness: float,
        coolant_fitness: float | None,
    ) -> Fitness:
        """
        Calculate and return a combined fitness based on electrical and thermal values.

        :param: electrical_fitness
            The current value of the electrical output/fitness.

        :param: thermal_fitness
            The current value of the thermal output/fitness.

        :param: coolant_fitness
            The current value of the coolant output/fitness.

        :returns: The weighted fitness.

        """

        return Fitness(
            (self.electrical_weighting / self.total_output_weighting)
            * electrical_fitness
            + (self.thermal_weighting / self.total_output_weighting) * thermal_fitness,
            electrical_fitness,
            thermal_fitness,
            coolant_fitness,
        )


class CollectorType(enum.Enum):
    """
    Denotes the type of collector being optimised.

    - PVT:
        A PV-T collector.

    - SSPVT:
        A spectrally-selective PV-T collector.

    """

    PVT = "pvt"
    SSPVT = "sspvt"


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
    ) -> tuple[float | None, float, float, dict[float, SystemData]]:
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
            - The coolant fitness of the model (if relevant),
            - The electrical fitness of the model,
            - The thermal fitness of the model.

        """

        # Make temporary files as needed based on the inputs for the run.
        with temporary_collector_file(
            self.base_pvt_filepath, self.date_and_time, kwargs, run_number
        ) as temp_collector_information:
            (
                temp_pvt_filepath,
                temp_pvt_collector_width,
                initial_pvt_collector_width,
            ) = temp_collector_information
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

        # Adjust the fitness attributes to cope with the whole collector
        segment_to_collector_scaling_factor = (
            initial_pvt_collector_width / temp_pvt_collector_width
        )

        # Use the run weights for each of the runs that were returned.
        electrical_fitness = (
            np.sum(entry.electrical_power for entry in output_data.values())
            * segment_to_collector_scaling_factor
        )
        thermal_fitness = np.sum(
            entry.thermal_power for entry in output_data.values()
        ) * int(segment_to_collector_scaling_factor)

        # Return these fitnesses.
        # NOTE: PV-T collectors of a non--spectrally splitting design have only one
        # thermal output, and, thus, the "coolant" fitness is `None`.
        return None, electrical_fitness, thermal_fitness, output_data

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
        (
            coolant_fitness,
            electrical_fitness,
            thermal_fitness,
            output_data,
        ) = self.unweighted_fitness_function(
            mass_flow_rate,
            run_number,
            solar_irradiance_data,
            temperature_data,
            wind_speed_data,
            **kwargs,
        )

        # Assess the fitness of the results and return.
        weighted_fitness = self.weighting_calculator.get_weighted_fitness(
            electrical_fitness, thermal_fitness, coolant_fitness
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
            print(
                "Negative fitness is "
                ", ".join(
                    [
                        str(entry)
                        for entry in (
                            electrical_fitness,
                            thermal_fitness,
                            weighted_fitness,
                        )
                        if entry < 0
                    ]
                )
            )

        return weighted_fitness


class SSPVTModelAssessor(CollectorModelAssessor, collector_type=CollectorType.SSPVT):
    """
    Class used for assessing the Gan Huang/SSPV-T model code.

    """

    def __init__(
        self,
        base_sspvt_filepath: str,
        base_model_input_files: list[str],
        date_and_time: DateAndTime,
        location_name: str,
        output_filename: str,
        *,
        weighting_calculator: WeightingCalculator,
    ) -> None:
        """
        Instnatiate the class.

        :param: base_sspvt_filepath
            The base SSPV-T filepath from which changes to the collector will be explored.

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

        self.base_sspvt_filepath: str = base_sspvt_filepath

        try:
            assert len(base_model_input_files) == 5
        except AssertionError:
            print("Incorrect number of input files for SSPVT run.")
            raise

        self.base_ambient_temperature_input_filepath = base_model_input_files[0]
        self.base_coolant_input_filepath = base_model_input_files[1]
        self.base_fluid_input_filepath = base_model_input_files[2]
        self.base_irradiance_input_filepath = base_model_input_files[3]
        self.base_wind_speed_input_filepath = base_model_input_files[4]

        self.date_and_time = date_and_time
        # self.model_args: list[str] = [
        #     "--skip-analysis",
        #     "--output",
        #     output_filename,
        #     "--decoupled",
        #     "--steady-state",
        #     "--initial-month",
        #     "7",
        #     "--location",
        #     os.path.join(INPUT_FILES_DIRECTORY, LOCATIONS_FOLDERNAME, location_name),
        #     "--portion-covered",
        #     "1",
        #     "--x-resolution",
        #     "31",
        #     "--y-resolution",
        #     "11",
        #     "--average-irradiance",
        #     "--skip-2d-output",
        #     "--layers",
        #     "g",
        #     "pv",
        #     "a",
        #     "p",
        #     "f",
        #     "--disable-logging",
        #     "--skip-output",
        # ]
        # self.output_filename = output_filename

        super().__init__(weighting_calculator)

    def _run_model(
        self, temporary_sspvt_filepath: str, suffix: str
    ) -> dict[float, Any]:
        """
        Run the model.

        :param: temporary_pvt_filepath
            The temporary filepath to the PV-T file to use.

        :param: temporary_steady_state_filepath
            The temporary filepath to the steady-state file to use.

        :returns:
            The results of the model.

        """

        # model_args = self.model_args + [
        #     "--steady-state-data-file",
        #     temporary_steady_state_filepath,
        # ]
        # model_args.extend(["--pvt-data-file", temporary_pvt_filepath])

        # Remove the output file if it already exists.
        # if os.path.isfile(self.output_filename):
        #     os.remove(self.output_filename)

        import pdb

        pdb.set_trace(header="_run_model")

        os.makedirs(SSPVT_BAYESIAN_OUTPUT_DIRECTORY, exist_ok=True)
        os.makedirs(
            os.path.join(HUANG_ET_AL_DIRECTORY, SSPVT_BAYESIAN_OUTPUT_DIRECTORY),
            exist_ok=True,
        )

        # Run the MATLAB model and return the outputs.
        # with Capturing():
        sspvt_model_main(panel_filename=temporary_sspvt_filepath, suffix=suffix)

        with open(
            os.path.join(
                SSPVT_BAYESIAN_OUTPUT_DIRECTORY,
                SSPVT_BAYESIAN_OUTPUT_FILENAME.format(
                    panel_filename=temporary_sspvt_filepath, suffix=suffix
                ),
            ),
            "r",
            encoding="UTF-8",
        ) as sspvt_bayesian_output_file:
            sspvt_bayesian_output = json.load(sspvt_bayesian_output_file)

        return sspvt_bayesian_output

    def unweighted_fitness_function(
        self,
        # mass_flow_rate: float,
        run_number: int,
        solar_irradiance_data: list[float],
        temperature_data: list[float],
        wind_speed_data: list[float],
        **kwargs,
    ) -> tuple[float | None, float, float, dict[float, SystemData]]:
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
            - The coolant fitness of the model (if relevant),
            - The electrical fitness of the model,
            - The thermal fitness of the model.

        """

        import pdb

        pdb.set_trace(header="SSPV-T model calling coming up")

        # Make temporary files as needed based on the inputs for the run.
        with temporary_collector_file(
            self.base_sspvt_filepath,
            self.date_and_time,
            kwargs,
            temp_upper_dirname=HUANG_ET_AL_DIRECTORY,
            unique_id=run_number,
        ) as temp_collector_information:
            # FIXME: Work from here :)
            (temp_sspvt_filepath, _, _) = temp_collector_information
            with temporary_sspvt_steady_state_files(
                self.base_ambient_temperature_input_filepath,
                self.base_coolant_input_filepath,
                self.base_fluid_input_filepath,
                self.base_irradiance_input_filepath,
                self.base_wind_speed_input_filepath,
                solar_irradiance_data,
                temperature_data,
                wind_speed_data,
                temp_upper_dirname=HUANG_ET_AL_DIRECTORY,
            ) as temp_steady_state_suffix:
                # Run the model.
                output_data = self._run_model(
                    temp_sspvt_filepath, temp_steady_state_suffix
                )

        import pdb

        pdb.set_trace(header="SSPV-T model run, fitness to be computed.")

        # Use the run weights for each of the runs that were returned.
        coolant_fitness = np.sum(entry.coolant_power for entry in output_data.values())
        electrical_fitness = np.sum(
            entry.electrical_power for entry in output_data.values()
        )
        thermal_fitness = np.sum(entry.thermal_power for entry in output_data.values())

        # Return these fitnesses.
        return coolant_fitness, electrical_fitness, thermal_fitness, output_data

    def fitness_function(
        self,
        # mass_flow_rate: float,
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
        (
            coolant_fitness,
            electrical_fitness,
            thermal_fitness,
            output_data,
        ) = self.unweighted_fitness_function(
            # mass_flow_rate,
            run_number,
            solar_irradiance_data,
            temperature_data,
            wind_speed_data,
            **kwargs,
        )

        # Assess the fitness of the results and return.
        weighted_fitness = self.weighting_calculator.get_weighted_fitness(
            electrical_fitness, thermal_fitness, coolant_fitness
        )

        _save_current_run(
            coolant_fitness=coolant_fitness,
            date_and_time=self.date_and_time,
            electrical_fitness=electrical_fitness,
            fitness=weighted_fitness,
            thermal_fitness=thermal_fitness,
            mass_flow_rate=None,
            run_number=run_number,
            **kwargs,
        )

        if (
            electrical_fitness < 0
            or thermal_fitness < 0
            or coolant_fitness < 0
            or weighted_fitness < 0
        ):
            print(
                "Negative fitness is "
                ", ".join(
                    [
                        str(entry)
                        for entry in (
                            coolant_fitness,
                            electrical_fitness,
                            thermal_fitness,
                            weighted_fitness,
                        )
                        if entry < 0
                    ]
                )
            )

        return weighted_fitness
