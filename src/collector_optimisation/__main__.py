#!/usr/bin/python3.10
########################################################################################
# main.py - Main module for the collector-optimisation module.                         #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2024                                                      #
########################################################################################

"""
The main module for the collector-optimisation software.

The main module is responsible for providing the entry point for the code.

"""

import argparse
import collections
import enum
import math
import matplotlib.pyplot as plt
import random
import os
import sys

from dataclasses import dataclass, field
from matplotlib import rc
from scipy.optimize import curve_fit
from typing import Any, Generator

import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from fast_pareto import is_pareto_front
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from .__utils__ import DateAndTime, INPUT_FILES_DIRECTORY, WeatherDataHeader
from .bayesian_optimiser import (
    BayesianPVTModelOptimiserSeries,
    BayesianPVTModelOptimiserThread,
    MAX_RESULTS_FILENAME,
)
from .model_wrapper import (
    CollectorModelAssessor,
    CollectorType,
    Fitness,
    RUNS_DATA_FILENAME,
    WeightingCalculator,
)


# COLLECTOR_FILES_DIRECTORY:
#   The name of the directory containing base collector files.
COLLECTOR_FILES_DIRECTORY: str = "collector_designs"

# LOCATIONS_FILENAME:
#   The name of the file containing the locations information.
LOCATIONS_FILENAME: str = "locations.yaml"

# MODEL_INPUTS_DIRECTORY:
#   The directory containing model inputs.
MODEL_INPUTS_DIRECTORY: str = "steady_state_data"

# OPTIMISATION_INPUTS_FILE
#   The name of the optimisations inputs file.
OPTIMISATION_INPUTS_FILE: str = "optimisation.yaml"

# PARAMETER_PRECISION_MAP:
#   Utilised for rounding the parameters based on the precision specified.
PARAMETER_PRECISION_MAP: dict[str, float] = {
    WeatherDataHeader.AMBIENT_TEMPERATURE: 1,
    WeatherDataHeader.SOLAR_IRRADIANCE: 1,
    WeatherDataHeader.WIND_SPEED: 1,
}

# PARETO_FRONT_FILENAME:
#    The filename for the Pareto front.
PARETO_FRONT_FILENAME: str = "pareto_front.yaml"

# SOLAR_FILENAME:
#   The name of the solar filename.
SOLAR_FILENAME: str = "ninja_pv_{lat:.4f}_{lon:.4f}_uncorrected.csv"

# THERMODYNAMIC_LIMIT:
#   A thermodynamic limit on efficiency placed on solar-thermal collectors.
THERMODYNAMIC_LIMIT: float = 1.0

# WEATHER_DIRECTORY:
#   The name of the directory containing the weather information.
WEATHER_DIRECTORY: str = "weather_data"

# WEATHER_SAMPLE_FILENAME:
#   The name of the weather sample file to use by default.
WEATHER_SAMPLE_FILENAME: str = "weather_data_sample"

# WIND_FILENAME:
#   The name of the wind filename.
WIND_FILENAME: str = "ninja_wind_{lat:.4f}_{lon:.4f}_corrected.csv"

# Seaborn setup
rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})
sns.set_context("notebook")
sns.set_style("whitegrid")
un_color_palette = sns.color_palette(
    [
        # "#C51A2E",
        # "#ED6A30",
        # "#FBC219",
        # "#2CBCE0",
        # "#2297D5",
        # "#0D699F",
        # "#19496A",
        "#8F1838",
        "#D9302F",
        "#F36D25",
        "#FDB713",
        "#00AED9",
        "#0169A4",
        "#183668",
    ]
)


@dataclass
class Location:
    """
    Represents a location.

    .. attribute:: name
        The name of the location.

    .. attribute:: latitude
        The latitude.

    .. attirbute:: longitude
        The longitude.

    """

    name: str
    lat: float
    lon: float

    @property
    def latitude(self) -> float:
        """Another name for the latitude."""

        return self.lat

    @property
    def longitude(self) -> float:
        """Another name for the longitude."""

        return self.lon

    def __eq__(self, other) -> bool:
        """Two locations are equal if they are at the same coordinates."""

        return (
            self.name == other.name
            and self.latitude == other.latitude
            and self.longitude == other.longitude
        )


class SampleType(enum.Enum):
    """
    Denotes the method to use for sampling.

    - DENSITY:
        Use a density approach.

    - GRID:
        Use a grid approach.

    """

    DENSITY = "density"
    GRID = "grid"


def _parse_args(args: list[Any]) -> argparse.Namespace:
    """
    Parses command-line arguments into a :class:`argparse.NameSpace`.

    :param: args
        The un-parsed command-line arguments.

    :returns: The parsed command-line arguments.

    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l", "--location", help="The name of the location to consider.", type=str
    )
    parser.add_argument(
        "-i",
        "--initial-points",
        default=16,
        help="The number of initial points to use.",
        type=int,
    )
    parser.add_argument(
        "-n",
        "--num-iterations",
        default=128,
        help="The number of iterations to carry out.",
        type=int,
    )
    parser.add_argument(
        "-po",
        "--plotting-only",
        action="store_true",
        default=False,
        help="Only run the plotting functionality.",
    )
    parser.add_argument(
        "-r",
        "--resample",
        action="store_true",
        default=False,
        help="When used, will resample weather data.",
    )
    parser.add_argument(
        "-wf",
        "--weather-sample-filename",
        default=WEATHER_SAMPLE_FILENAME,
        help="The name of the weather-sample file to use when saving and loading data.",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weather-sample-size",
        help="The number of weather points to sample.",
        type=int,
    )

    collector_model_args = parser.add_argument_group(
        "Collector-model files",
        "Parameters needed to specify files utilised in the underlying models that are "
        "being optimised.",
    )
    collector_model_args.add_argument(
        "-bc",
        "--base-collector-filename",
        default=None,
        help="The name of the base collector file.",
        type=str,
    )
    collector_model_args.add_argument(
        "-bmfs",
        "--base-model-input-files",
        help="A list of base model-inputs files. Must be specified in order, see "
        "documentation for details.",
        nargs="+",
    )

    plotting_only_args = parser.add_argument_group(
        "Plotting-only arguments",
        "Parameters that can be used when plotting to load data for a specific date.",
    )
    plotting_only_args.add_argument(
        "-d",
        "--date",
        default=None,
        help="The date to use when loading results for plotting.",
    )
    plotting_only_args.add_argument(
        "-t",
        "--time",
        default=None,
        help="The time to use when loading resiult for plotting.",
    )

    return parser.parse_args(args)


def _parse_files(
    base_collector_filename: str,
    base_model_input_files: list[str],
    date_and_time: DateAndTime,
    location_name: str,
    *,
    resample: bool = False,
    sample_type: SampleType = SampleType.DENSITY,
    weather_sample_filename: str,
    weather_sample_size: int = 40,
) -> tuple[list[CollectorModelAssessor], pd.DataFrame, pd.DataFrame]:
    """
    Parse the input files needed to run the model.

    :param: base_collector_filename
        The base collector filename on which adjustments are made.

    :param: base_model_input_files
        A `list` of filenames which go into the underlying model.

    :param: date_and_time
        The date and time to use when saving files.

    :param: location_name
        The name of the location to consider.

    :param: resample
        If specified, the weather data will be resampled.

    :param: sample_type
        The type of sampling to use.

    :param: weather_sample_filename
        The name of the weather-sample file.

    :param: weather_sample_size
        The sample size to use when sampling the weather data.

    :returns:
        - The collector model assessors;
        - The weather data to run;
        - The complete weather data.

    """

    # Parse the location inputs.
    with open(
        os.path.join(INPUT_FILES_DIRECTORY, LOCATIONS_FILENAME), "r", encoding="UTF-8"
    ) as locations_file:
        locations = [Location(**entry) for entry in yaml.safe_load(locations_file)]

    # Attempt to determine the location.
    try:
        location = [
            location for location in locations if location.name == location_name
        ][0]
    except IndexError:
        raise

    def _add_csv(filepath: str) -> str:
        if filepath.endswith(".csv"):
            return filepath
        return f"{filepath}.csv"

    # Parse the solar and wind files for the location specified.
    with open(
        os.path.join(
            INPUT_FILES_DIRECTORY,
            WEATHER_DIRECTORY,
            SOLAR_FILENAME.format(lat=location.lat, lon=location.lon),
        ),
        "r",
        encoding="UTF-8",
    ) as solar_file:
        solar_data = pd.read_csv(solar_file, comment="#")

    with open(
        os.path.join(
            INPUT_FILES_DIRECTORY,
            WEATHER_DIRECTORY,
            WIND_FILENAME.format(lat=location.lat, lon=location.lon),
        ),
        "r",
        encoding="UTF-8",
    ) as wind_file:
        wind_data = pd.read_csv(wind_file, comment="#")

    # Combine the two dataframes
    combined_weather_data = solar_data
    combined_weather_data["wind_speed"] = wind_data["wind_speed"]
    combined_weather_data["irradiance_total"] = 1000 * (
        combined_weather_data["irradiance_diffuse"]
        + combined_weather_data["irradiance_direct"]
    )

    # Sample the weather data using one of two methods
    modelling_weather_data = pd.DataFrame(
        combined_weather_data[combined_weather_data["irradiance_total"] > 0]
    )
    modelling_weather_array = modelling_weather_data.loc[
        :, ["irradiance_total", "temperature", "wind_speed"]
    ].to_numpy()
    # modelling_weather_data["month"] = [int(entry.split("-")[1]) for entry in modelling_weather_data["time"]]

    column_headers = [
        WeatherDataHeader.SOLAR_IRRADIANCE.value,
        WeatherDataHeader.AMBIENT_TEMPERATURE.value,
        WeatherDataHeader.WIND_SPEED.value,
    ]

    # Sample the weather data if requested or required
    weather_sample_filepath = _add_csv(
        os.path.join(INPUT_FILES_DIRECTORY, WEATHER_DIRECTORY, WEATHER_SAMPLE_FILENAME)
    )

    if resample or not os.path.isfile(weather_sample_filepath):

        def _density_based_sampling(
            data: np.ndarray, sample_size: int, bandwidth: float = 0.2
        ) -> np.ndarray:
            """
            Performs density-based sampling on a given dataset.

            :param: The input data as a NumPy array.

            :param: sample_size
                The desired sample size.

            Returns:
                A NumPy array of sampled points.

            """

            # Kernel Density Estimation
            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
            kde.fit(data)

            # Calculate density scores
            density_scores = np.exp(kde.score_samples(data))

            # Normalize density scores to probabilities
            probabilities = density_scores / np.sum(density_scores)

            # Sample points based on probabilities
            indices = np.random.choice(
                len(data), size=sample_size, p=probabilities, replace=False
            )
            sample = data[indices]

            return sample

        def _weighted_grid_sampling(
            data: np.ndarray, grid_size: int | list[int], sample_size: int
        ) -> np.ndarray:
            """
            Performs grid sampling with weighted selection based on point density.

            :param: data
                The input data as a NumPy array.

            :param: grid_size
                The size of the grid cells.

            :param: sample_size
                The desired sample size.

            Returns:
                A NumPy array of sampled points.

            """

            # Determine grid boundaries
            min_values = np.floor(np.min(data, axis=0))
            max_values = np.ceil(np.max(data, axis=0))
            grid_ranges = max_values - min_values

            # Create a grid of cells
            if isinstance(grid_size, int):
                grid_shape = (
                    np.floor(grid_ranges[0] / grid_size),
                    np.floor(grid_ranges[1] / grid_size),
                    np.floor(grid_ranges[2] / grid_size),
                )
            else:
                grid_shape = (
                    np.floor(grid_ranges[0] / grid_size[0]),
                    np.floor(grid_ranges[1] / grid_size[1]),
                    np.floor(grid_ranges[2] / grid_size[2]),
                )

            grid = np.zeros([int(entry) + 1 for entry in grid_shape], dtype=int)
            grid_to_cells: collections.defaultdict[
                tuple[int, int, int], list[tuple[float, float, float]]
            ] = collections.defaultdict(list)

            # Assign points to grid cells and store
            for point in data:
                cell_indices = np.floor((point - min_values) / grid_size).astype(int)
                grid[(grid_coordinates := tuple(cell_indices))] += 1
                grid_to_cells[grid_coordinates].append(point)

            # Flatten grid to a list of (cell index, count) pairs
            grid_flat = np.argwhere(grid > 0)
            grid_counts = grid[grid_flat[:, 0], grid_flat[:, 1], grid_flat[:, 2]]

            # Calculate probabilities based on counts
            probabilities = grid_counts / np.sum(grid_counts)

            # Sample grid cells based on probabilities
            selected_cells = np.random.choice(
                len(grid_flat), size=sample_size, p=probabilities, replace=False
            )

            # Select random points from selected cells
            sample = []
            for cell_index in selected_cells:
                grid_coordinates = tuple(grid_flat[cell_index])
                cell_points = grid_to_cells[grid_coordinates]
                sample.append(random.choice(cell_points))

            return np.array(sample)

        # Sample until the user is happy with the sample.
        while input("Is the weather sample sufficient? Yes [y] or no [n]? ") not in (
            "Yes",
            "yes",
            "Y",
            "y",
        ):
            density_based_weather_sample = _density_based_sampling(
                modelling_weather_array, weather_sample_size, bandwidth=0.4
            )
            grid_based_weather_sample = _weighted_grid_sampling(
                modelling_weather_array, [50, 5, 1], weather_sample_size
            )

            # Return the sampled weather data alone with the model assessors.
            weather_sample = pd.DataFrame(
                density_based_weather_sample
                if sample_type == SampleType.DENSITY
                else grid_based_weather_sample
            )
            weather_sample.columns = pd.Index((column_headers))

            # Plot and display to the user.
            sns.set_context("notebook")
            sns.set_style("ticks")

            sns.set_palette(
                [
                    "#2CBCE0",
                    "#0D699F",
                ]
            )

            sns.jointplot(
                modelling_weather_data,
                x="irradiance_total",
                y="temperature",
                marker="h",
                alpha=0.2,
                linewidth=0,
                height=32 / 5,
                ratio=4,
                marginal_kws={"bins": 20},
            )
            ax = plt.gca()
            ax.set_xlabel("Irradiance / W/m$^2$")
            ax.set_ylabel("Temperature / $^\circ$C")

            plt.scatter(
                x=weather_sample.loc[:, WeatherDataHeader.SOLAR_IRRADIANCE.value],
                y=weather_sample.loc[:, WeatherDataHeader.AMBIENT_TEMPERATURE.value],
                s=100,
                marker="H",
                alpha=0.8,
                linewidth=0,
                color="C1",
            )

            plt.show()

        print("Sample approved, continuing...")

        with open(
            weather_sample_filepath, "w", encoding="UTF-8"
        ) as weather_sample_file:
            weather_sample.to_csv(weather_sample_file, index=None)

    else:
        with open(
            weather_sample_filepath, "r", encoding="UTF-8"
        ) as weather_sample_file:
            weather_sample = pd.read_csv(weather_sample_file)

    ###################################################################
    # Code for plotting and visualising the weather-data distribution #
    ###################################################################

    sns.set_context("notebook")
    sns.set_style("ticks")

    sns.set_palette(
        [
            "#2CBCE0",
            "#0D699F",
        ]
    )

    sns.jointplot(
        modelling_weather_data,
        x="irradiance_total",
        y="temperature",
        marker="h",
        alpha=0.2,
        linewidth=0,
        height=32 / 5,
        ratio=4,
        marginal_kws={"bins": 20},
    )
    ax = plt.gca()
    ax.set_xlabel("Irradiance / W/m$^2$")
    ax.set_ylabel("Temperature / $^\circ$C")

    plt.scatter(
        x=weather_sample.loc[:, WeatherDataHeader.SOLAR_IRRADIANCE.value],
        y=weather_sample.loc[:, WeatherDataHeader.AMBIENT_TEMPERATURE.value],
        s=100,
        marker="H",
        alpha=0.8,
        linewidth=0,
        color="C1",
    )
    plt.savefig(
        f"scatter_plot_{sample_type.value}_weather_{date_and_time.date}_{date_and_time.time}.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )

    # sns.jointplot(
    #     modelling_weather_data,
    #     x="irradiance_total",
    #     y="temperature",
    #     marker="h",
    #     alpha=0.2,
    #     linewidth=0,
    #     height=32 / 5,
    #     ratio=4,
    #     marginal_kws={"bins": 20},
    # )
    # ax = plt.gca()
    # ax.set_xlabel("Irradiance / W/m$^2$")
    # ax.set_ylabel("Temperature / $^\circ$C")

    # plt.scatter(
    #     x=grid_based_weather_sample[:, 0],
    #     y=grid_based_weather_sample[:, 1],
    #     s=100,
    #     marker="H",
    #     alpha=0.8,
    #     linewidth=0,
    #     color="C1",
    # )
    # plt.savefig("scatter_plot_grid_weather.pdf", bbox_inches="tight", pad_inches=0)

    sns.jointplot(
        modelling_weather_data,
        x="irradiance_total",
        y="temperature",
        linewidth=0,
        height=32 / 5,
        ratio=4,
        marginal_kws={"bins": 20},
        kind="hex",
    )
    ax = plt.gca()
    ax.set_xlabel("Irradiance / W/m$^2$")
    ax.set_ylabel("Temperature / $^\circ$C")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.savefig(
        f"hex_plot_weather_data_{date_and_time.date}_{date_and_time.time}.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )

    # modelling_weather_data["rounded_wind_speed"] = [
    #     int(entry) for entry in modelling_weather_data["wind_speed"]
    # ]
    # fig, axes = plt.subplots(4, 4, figsize=(48 / 5, 48 / 5))
    # fig.subplots_adjust(hspace=0.73, wspace=0.45)
    # flat_axes = np.argwhere(axes)
    # sns.set_palette(sns.cubehelix_palette(start=0.6, rot=-0.4))
    # for index, rounded_wind_speed in enumerate(
    #     sorted(set(modelling_weather_data["rounded_wind_speed"]))
    # ):
    #     try:
    #         sns.histplot(
    #             modelling_weather_data[
    #                 modelling_weather_data["rounded_wind_speed"] == rounded_wind_speed
    #             ],
    #             x="irradiance_total",
    #             y="temperature",
    #             ax=(ax := axes[*flat_axes[index]]),
    #             bins=20,
    #         )
    #         # grid = sns.jointplot(
    #         #     modelling_weather_data[modelling_weather_data["rounded_wind_speed"] == rounded_wind_speed],
    #         #     x="irradiance_total",
    #         #     y="temperature",
    #         #     marker="h",
    #         #     alpha=0.2,
    #         #     linewidth=0,
    #         #     height=32 / 5,
    #         #     ratio=4,
    #         #     marginal_kws={"bins": 20},
    #         #     kind="scatter",
    #         # )
    #     except ZeroDivisionError:
    #         continue
    #     ax.set_xlim(*xlim)
    #     ax.set_ylim(*ylim)
    #     ax.set_xlabel("Irradiance / W/m$^2$")
    #     ax.set_ylabel("Temperature / $^\circ$C")
    #     ax.set_title(f"Wind speed = {rounded_wind_speed} m/s")

    # plt.savefig("hist_plot_wind_speeds.pdf", bbox_inches="tight", pad_inches=0)

    # Parse the pareto-front informtion and return wrappers based on this.
    with open(
        os.path.join(INPUT_FILES_DIRECTORY, PARETO_FRONT_FILENAME),
        "r",
        encoding="UTF-8",
    ) as pareto_front_file:
        pareto_front_data = yaml.safe_load(pareto_front_file)

    collector_model_assessors: list[CollectorModelAssessor] = []
    for model_name, pareto_front_parameters in pareto_front_data.items():
        # Determing the weighting functions.
        try:
            weighting_calculators = [
                WeightingCalculator(**pareto_front_data)
                for pareto_front_data in pareto_front_parameters
            ]
        # A TypeError is raised if the parameters in the pareto front file don't match
        # those needed to instantiate a weighting point on the Pareto front.
        except TypeError:
            raise

        # Instantiate the model assessors.
        collector_model_assessors.extend(
            [
                CollectorModelAssessor.collector_type_to_wrapper[
                    CollectorType(model_name)
                ](
                    base_collector_filename,
                    base_model_input_files,
                    date_and_time,
                    location.name,
                    f"{model_name}_with_{weighting_calculator.name}",
                    weighting_calculator=weighting_calculator,
                )
                for weighting_calculator in weighting_calculators
            ]
        )

    # Parse the optimisation inputs information and convert into a format that is usable
    # by the Bayesian optimisation script.
    with open(
        os.path.join(INPUT_FILES_DIRECTORY, OPTIMISATION_INPUTS_FILE),
        "r",
        encoding="UTF-8",
    ) as optimisation_inputs_file:
        optimisation_inputs = yaml.safe_load(optimisation_inputs_file)

    optimisation_parameters = {
        key: (value["min"], value["max"]) for key, value in optimisation_inputs.items()
    }

    weather = pd.DataFrame(modelling_weather_array)
    weather.columns = pd.Index(column_headers)

    return collector_model_assessors, optimisation_parameters, weather_sample, weather


def _validate_args(parsed_args: argparse.Namespace) -> tuple[str, list[str]]:
    """
    Raises errors if the parsed args aren't valid arguments.

    :param: parsed_args
        The parsed command-line arguments.

    :raises: Exception
        Raised if the arguments aren't correct.
    :raises: FileNotFoundError
        Raised if the files can't be found that are expected.

    :returns:
        - The base collector filepath.
        - The list of base input files.

    """

    # Check that the collector filepath exists.
    try:
        if not os.path.isfile(
            (
                base_collector_filepath := os.path.join(
                    INPUT_FILES_DIRECTORY,
                    COLLECTOR_FILES_DIRECTORY,
                    parsed_args.base_collector_filename,
                )
            )
        ):
            raise FileNotFoundError(
                f"The base collector file '{base_collector_filepath}' could not be found in the expected directory, {os.path.dirname(base_collector_filepath)}"
            )
    except TypeError:
        raise Exception(
            "The base collector filepath must be specified on the CLI."
        ) from None

    # Check that each file exists that is a base input to the model.
    try:
        base_model_input_filepaths = [
            os.path.join(INPUT_FILES_DIRECTORY, MODEL_INPUTS_DIRECTORY, filename)
            for filename in parsed_args.base_model_input_files
        ]
    except TypeError:
        raise Exception(
            "Base model input files must be specified on the CLI."
        ) from None

    for filepath in base_model_input_filepaths:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"The base model inputs file {os.path.basename(filepath)} could not be found."
            )

    return base_collector_filepath, base_model_input_filepaths


def plot_pareto_front(
    date_and_time: DateAndTime,
    optimisation_parameters: dict[Any, tuple[Any, Any]],
    weather_data_sample: pd.DataFrame,
    *,
    electrical_weightings: dict[int, float] | None = None,
    num_repeats: int = 1,
    thermal_weightings: dict[int, float] | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """
    Plots the Pareto front.

    :param: date_and_time
        The date and time to use when parsing the information.

    :param: optimisation_parameters
        The optimisation parameters

    :param: werather_data_sample
        The weather-data sample to use.

    :param: electrical_weightings
        The electrical weightings to use, as a `dict` instance.

    :param: num_repeats
        The number of steady-state runs that took place, and so how many additional runs
        took place for each data point.

    :param: thermal_weightings
        The thermal weightings to use, as a `dict` instance.

    """

    # Ensure all previous plots have closed.
    plt.close()

    # Open the data and parse the values.
    try:
        with open(
            MAX_RESULTS_FILENAME.format(
                date=date_and_time.date, time=date_and_time.time
            ),
            "r",
        ) as max_results_file:
            max_results: pd.DataFrame | None = pd.read_csv(max_results_file)
    except FileNotFoundError:
        max_results = None

    with open(
        RUNS_DATA_FILENAME.format(date=date_and_time.date, time=date_and_time.time), "r"
    ) as runs_data_file:
        runs_data = pd.read_csv(runs_data_file)

    if max_results is not None:
        maximal_runs = pd.DataFrame(
            [
                row[1]
                for row in runs_data.iterrows()
                if row[1]["fitness"] in max_results["target"].values
            ]
        )

        maximal_runs = maximal_runs.sort_values(by="run_number")

        # plt.plot(maximal_runs["thermal_fitness"], maximal_runs["electrical_fitness"], "--", color="grey")
        def pareto_function(x, a, b, c, d) -> float:
            return b * np.sqrt(1 - (x - d) ** 2 / a**2) + c

        try:
            parameters, _ = curve_fit(
                pareto_function,
                maximal_runs["thermal_fitness"],
                maximal_runs["electrical_fitness"],
                p0=[
                    maximal_runs["thermal_fitness"].max(),
                    maximal_runs["electrical_fitness"].max(),
                    0,
                    0,
                ],
                maxfev=10000,
            )

        except RuntimeError:
            pass
        else:
            plt.plot(
                (
                    pareto_thermal := np.linspace(
                        maximal_runs["thermal_fitness"].min(),
                        maximal_runs["thermal_fitness"].max(),
                        1000,
                    )
                ),
                pareto_function(pareto_thermal, *parameters),
                "--",
                color="grey",
                label="Fitted Pareto front",
            )

    else:
        maximal_runs = None

    # Normalise the runs data based on the total input values
    cumulative_solar_irradiance = np.sum(
        weather_data_sample[WeatherDataHeader.SOLAR_IRRADIANCE.value]
    )
    # average_solar_irradiance = cumulative_solar_irradiance / len(weather_data_sample)
    max_collector_size = 1.4276
    # max_collector_width = optimisation_parameters["pvt_collector/width"][1]
    max_electrical_efficiency = optimisation_parameters["pv/reference_efficiency"][1]

    energy_input = cumulative_solar_irradiance * num_repeats * max_collector_size
    runs_data["normalised_electrical_fitness"] = runs_data["electrical_fitness"] / (
        max_electrical_efficiency * energy_input
    )
    runs_data["normalised_thermal_fitness"] = runs_data["thermal_fitness"] / (
        THERMODYNAMIC_LIMIT * energy_input
    )

    # Clip the runs data to be in-bounds
    runs_data = runs_data[
        (runs_data["electrical_fitness"] > 0)
        & (runs_data["normalised_thermal_fitness"] > 0)
    ]

    electrical_values = np.linspace(0, 1, 100)
    thermal_values = [1 - entry for entry in electrical_values]

    # Write normalised legend labels
    summed_weightings = {
        key: thermal_weightings[key] + value
        for key, value in electrical_weightings.items()
    }

    def _normalise_weightings(run_index: int, weighting: float) -> float:
        """
        Normalise a weighting so that all summed weightings are the same.

        :param: run_index
            The index for the run.

        :param: weighting
            The weighting to normalise.

        :returns:
            The normalised weighting.

        """
        return (
            weighting * max(summed_weightings.values()) / summed_weightings[run_index]
        )

    electrical_weightings = {
        key: _normalise_weightings(key, value)
        for key, value in electrical_weightings.items()
    }
    thermal_weightings = {
        key: _normalise_weightings(key, value)
        for key, value in thermal_weightings.items()
    }
    labels: dict[int, str] = {
        run_number: rf"w$_\mathrm{{th}}$={thermal_weightings[run_number]:.0f}, w$_\mathrm{{el}}$={electrical_weightings[run_number]:.0f}"
        for run_number in range(len(electrical_weightings))
    }

    # Setup a new figure for the Parety front.
    plt.figure(figsize=(48 / 5, 32 / 5))

    sns.scatterplot(
        runs_data[
            (runs_data["electrical_fitness"] > 0) & (runs_data["thermal_fitness"] > 0)
        ],
        x="normalised_thermal_fitness",
        y="normalised_electrical_fitness",
        # color="grey",
        hue="run_number",
        palette=un_color_palette,
        marker="h",
        s=200,
        alpha=0.15,
        linewidth=0,
        legend=False,
    )

    # Plot the max-results flie if present, otherwise plot the maximal results from
    # those obtained.
    if max_results is not None:
        maximal_runs["normalised_electrical_fitness"] = maximal_runs[
            "electrical_fitness"
        ] / (max_electrical_efficiency * energy_input)
        maximal_runs["normalised_thermal_fitness"] = maximal_runs["thermal_fitness"] / (
            THERMODYNAMIC_LIMIT * energy_input
        )
    else:
        maximal_runs = pd.DataFrame(
            {
                index: runs_data[
                    (runs_data["run_number"] == index)
                    & (runs_data["electrical_fitness"] > 0)
                    & (runs_data["thermal_fitness"] > 0)
                ]
                .sort_values("fitness", ascending=False)
                .iloc[0]
                for index in set(runs_data["run_number"])
            }
        ).transpose()
        maximal_runs["run_number"] = maximal_runs["run_number"].astype(int)

    sns.scatterplot(
        maximal_runs,
        x="normalised_thermal_fitness",
        y="normalised_electrical_fitness",
        hue="run_number",
        palette=un_color_palette,
        s=200,
        alpha=1.0,
        marker="h",
    )

    # Determine and plot the Pareto front
    pareto_front_frame = runs_data[
        is_pareto_front(
            runs_data.loc[
                :,
                ["normalised_electrical_fitness", "normalised_thermal_fitness"],
            ].values,
            larger_is_better_objectives=[0, 1],
        )
    ]
    # sns.scatterplot(
    #     pareto_front_frame,
    #     x="normalised_thermal_fitness",
    #     y="normalised_electrical_fitness",
    #     hue="run_number",
    #     palette=un_color_palette,
    #     s=200,
    #     marker="h",
    # )
    sns.lineplot(
        pareto_front_frame,
        x="normalised_thermal_fitness",
        y="normalised_electrical_fitness",
        dashes=(4, 2),
        color="grey",
    )

    # plt.plot(thermal_values, electrical_values, "--", label="Maximum obtainable power")
    plt.xlabel(
        r"Normalised thermal energy produced ($f_\mathrm{th}$) / kWh$_\mathrm{th}$/kWh$_\mathrm{in}$"
    )
    plt.ylabel(
        r"Normalised electrical energy produced ($f_\mathrm{el}$) / kWh$_\mathrm{el}$/kWh$_\mathrm{in}$"
    )
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels.values())
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # handles, _ = (axis := plt.gca()).get_legend_handles_labels()
    # plt.legend(handles, labels)

    plt.savefig(
        f"pareto_front_{date_and_time.date}_{date_and_time.time}.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.show()

    def _generate_tangent_points(
        point: tuple[float, float],
        gradient: float,
        num_points: int = 100,
        x_min: float = 0,
        x_max: float = 1,
        y_min: float = 0,
        y_max: float = 1,
    ) -> list[tuple[float, float]]:
        """
        Generates points for plotting a Pareto tangent.

        :param: point
            A `tuple` which represents the point.

        :param: gradient
            The gradient of the line at the point.

        :param: num_points
            The number of points to plot.

        """

        # For an infinite gradient, plot a vertical line.
        if abs(gradient) == np.inf:
            y_values = np.linspace(y_min, y_max, num_points)
            x_values = [point[0]] * len(y_values)

        # For a zero gradient, plot a horizontal line.
        elif gradient == 0:
            x_values = np.linspace(x_min, x_max, num_points)
            y_values = [point[1]] * len(x_values)

        # Otherwise, plot a y=mx+c line.
        else:
            x_values = np.linspace(x_min, x_max, num_points)
            y_values = gradient * (x_values - point[0]) + point[1]

        return list(zip(x_values, y_values))

    # Plot the Pareto front with subplots for each of the run numbers.
    def subplots_label() -> Generator[str, None, None]:
        index = 0
        labels = [f"{entry}." for entry in ["a", "b", "c", "d", "e", "f", "g", "h"]]

        while index in range(len(labels)):
            yield labels[index]

            index += 1

        return StopIteration()

    fig, axes = plt.subplots(3, 3, figsize=(48 / 5, 48 / 5))
    fig.subplots_adjust(hspace=0.55, wspace=0.45)

    sns.set_palette(un_color_palette)
    subplots_labels = subplots_label()

    for run_number in sorted(set(runs_data["run_number"])):
        axis = axes[run_number // 3, run_number % 3]
        # Plot all points, coloured only if the run number matches.
        sns.scatterplot(
            runs_data[
                (runs_data["electrical_fitness"] > 0)
                & (runs_data["thermal_fitness"] > 0)
                & (runs_data["run_number"] != run_number)
            ],
            x="normalised_thermal_fitness",
            y="normalised_electrical_fitness",
            ax=axis,
            color="lightgrey",
            # hue="run_number",
            # palette=un_color_palette,
            marker="h",
            s=200,
            alpha=0.05,
            linewidth=0,
            legend=False,
        )
        sns.scatterplot(
            runs_data[
                (runs_data["electrical_fitness"] > 0)
                & (runs_data["thermal_fitness"] > 0)
                & (runs_data["run_number"] == run_number)
            ],
            x="normalised_thermal_fitness",
            y="normalised_electrical_fitness",
            ax=axis,
            color=f"C{run_number}",
            # hue="run_number",
            # palette=un_color_palette,
            marker="h",
            s=200,
            alpha=0.25,
            linewidth=0,
            legend=False,
        )
        sns.scatterplot(
            (maximal_run := maximal_runs[maximal_runs["run_number"] == run_number]),
            x="normalised_thermal_fitness",
            y="normalised_electrical_fitness",
            ax=axis,
            color=f"C{run_number}",
            s=200,
            alpha=1.0,
            marker="h",
        )
        tangent_line_points = _generate_tangent_points(
            (
                (
                    normalised_thermal_fitness_point := float(
                        maximal_run["normalised_thermal_fitness"].iloc[0]
                    )
                ),
                float(maximal_run["normalised_electrical_fitness"].iloc[0]),
            ),
            (
                gradient := -np.divide(
                    thermal_weightings[run_number],
                    electrical_weightings[run_number] * (max_electrical_efficiency),
                )
            ),
        )
        line = sns.lineplot(
            x=[entry[0] for entry in tangent_line_points],
            y=[entry[1] for entry in tangent_line_points],
            ax=axis,
            color=f"C{run_number}",
            dashes=(2, 2),
            label="Optimisation front",
        )
        if run_number == 0:
            axis.axvline(x=normalised_thermal_fitness_point, dashes=(2, 2))
        axis.legend().remove()
        sns.despine(offset=10)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel(None)
        axis.set_ylabel(None)
        axis.set_title(
            rf"w$_\mathrm{{th}}$={thermal_weightings[run_number]:.0f}, w$_\mathrm{{el}}$={electrical_weightings[run_number]:.0f}",
            fontweight="bold",
        )
        axis.text(-0.25, 1.1125, next(subplots_labels), fontsize=16, fontweight="bold")

    axes[2, 1].set_visible(False)
    axes[2, 2].set_visible(False)

    fig.text(
        0.5,
        0.04,
        r"Normalised thermal energy produced ($f_\mathrm{th}$) / kWh$_\mathrm{th}$/kWh$_\mathrm{in}$",
        ha="center",
        va="center",
    )
    fig.text(
        0.04,
        0.5,
        r"Normalised electrical energy produced ($f_\mathrm{el}$) / kWh$_\mathrm{el}$/kWh$_\mathrm{in}$",
        ha="center",
        va="center",
        rotation="vertical",
    )
    fig.legend(
        [line],
        ["Optimisation front(s)"],
        loc="lower right",
    )

    plt.savefig(
        f"pareto_subplots_{date_and_time.date}_{date_and_time.time}.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.show()

    # Plot the variation of the design parameters across the Pareto front.
    design_variables = [
        entry
        for entry in runs_data.columns
        if "fitness" not in entry and entry != "run_number"
    ]
    variable_labels: dict[str, str] = {
        "mass_flow_rate": "Mass flow rate / litres/hour",
        "air_gap/thickness": "Air-gap thickness / m",
        "glass/absorptivity": r"Glass apsorptivity ($\alpha_\mathrm{g}$)",
        "glass/emissivity": r"Glass emissivity ($\epsilon_\mathrm{g}$)",
        "pv/reference_efficiency": r"PV cell reference efficiency ($\eta_\mathrm{el,ref}$)",
        "pv/thermal_coefficient": r"PV thermal coefficient ($\beta_\mathrm{pv}$)",
        "pvt_collector/width": r"Pipe spacing / m",
    }

    for variable in tqdm(design_variables, desc="Plotting design KDEs", leave=True):
        # Setup a new figure for the KDE analysis front.
        joint_plot_grid = sns.jointplot(
            (kde_frame := pd.concat([pareto_front_frame, maximal_runs])),
            x=variable,
            y="normalised_electrical_fitness",
            hue="run_number",
            palette=un_color_palette,
            alpha=1.0,
            kind="scatter",
            marker="h",
            s=200,
            # linewidths=2,
            zorder=1,
            marginal_ticks=False,
            ratio=3,
        )
        sns.scatterplot(
            runs_data[
                (runs_data["electrical_fitness"] > 0)
                & (runs_data["thermal_fitness"] > 0)
            ],
            x=variable,
            y="normalised_electrical_fitness",
            # color="grey",
            hue="run_number",
            palette=un_color_palette,
            marker="h",
            s=200,
            alpha=0.05,
            linewidth=0,
            legend=False,
            ax=joint_plot_grid.ax_joint,
            zorder=0,
        )
        # joint_plot_grid.plot_marginals(sns.rugplot, height=-.15, clip_on=False)
        plt.xlabel(variable_labels[variable])
        plt.ylabel(
            r"Normalised electrical energy produced ($f_\mathrm{el}$) / kWh$_\mathrm{el}$/kWh$_\mathrm{in}$"
        )
        plt.ylim(0, 1)
        handles, plotted_labels = (axis := plt.gca()).get_legend_handles_labels()
        plt.legend(
            handles[:7], [labels[int(plotted_label)] for plotted_label in range(7)]
        )
        # sns.despine(offset=10)
        joint_plot_grid.ax_marg_x.tick_params(axis="x", left=False, labelleft=False)
        joint_plot_grid.ax_marg_x.set_xlabel("Average irradiance / kWm$^{-2}$")
        joint_plot_grid.ax_marg_y.tick_params(axis="y", bottom=False, labelbottom=False)
        plt.savefig(
            f"optimum_electrical_with_kde_{variable.replace('/', '_')}.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

        # Setup a new figure for the KDE analysis front.
        joint_plot_grid = sns.jointplot(
            (kde_frame := pd.concat([pareto_front_frame, maximal_runs])),
            x=variable,
            y="normalised_thermal_fitness",
            hue="run_number",
            palette=un_color_palette,
            alpha=1.0,
            kind="scatter",
            marker="h",
            s=200,
            # linewidths=2,
            zorder=1,
            marginal_ticks=False,
            # ratio=3
        )
        sns.scatterplot(
            runs_data[
                (runs_data["electrical_fitness"] > 0)
                & (runs_data["thermal_fitness"] > 0)
            ],
            x=variable,
            y="normalised_thermal_fitness",
            # color="grey",
            hue="run_number",
            palette=un_color_palette,
            marker="h",
            s=200,
            alpha=0.05,
            linewidth=0,
            legend=False,
            ax=joint_plot_grid.ax_joint,
            zorder=0,
        )
        # joint_plot_grid.plot_marginals(sns.rugplot, height=-.15, clip_on=False)
        plt.xlabel(variable_labels[variable])
        plt.ylabel(
            r"Normalised thermal energy produced ($f_\mathrm{th}$) / kWh$_\mathrm{th}$/kWh$_\mathrm{in}$"
        )
        plt.ylim(0, 1)
        handles, plotted_labels = (axis := plt.gca()).get_legend_handles_labels()
        plt.legend(
            handles[:7], [labels[int(plotted_label)] for plotted_label in range(7)]
        )
        # sns.despine(offset=10)
        joint_plot_grid.ax_marg_x.tick_params(axis="x", left=False, labelleft=False)
        joint_plot_grid.ax_marg_x.set_xlabel("Average irradiance / kWm$^{-2}$")
        joint_plot_grid.ax_marg_y.tick_params(axis="y", bottom=False, labelbottom=False)
        plt.savefig(
            f"optimum_thermal_with_kde_{variable.replace('/', '_')}.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

    plt.show()

    for variable in tqdm(design_variables, desc="Plotting design impact", leave=True):

        # Plot the electrical fitness with only maximal values.
        plt.figure(figsize=(48 / 5, 32 / 5))

        sns.scatterplot(
            runs_data[
                (runs_data["electrical_fitness"] > 0)
                & (runs_data["thermal_fitness"] > 0)
            ],
            x=variable,
            y="normalised_electrical_fitness",
            # color="grey",
            hue="run_number",
            palette=un_color_palette,
            marker="h",
            s=200,
            alpha=0.05,
            linewidth=0,
            legend=False,
        )
        sns.scatterplot(
            maximal_runs,
            x=variable,
            y="normalised_electrical_fitness",
            hue="run_number",
            palette=un_color_palette,
            s=200,
            alpha=1.0,
            marker="h",
        )

        plt.xlabel(variable_labels[variable])
        plt.ylabel(
            r"Normalised electrical energy produced ($f_\mathrm{el}$) / kWh$_\mathrm{el}$/kWh$_\mathrm{in}$"
        )
        plt.ylim(0, 1)

        handles, plotted_labels = (axis := plt.gca()).get_legend_handles_labels()
        plt.legend(handles, [labels[int(label)] for label in plotted_labels])

        plt.savefig(
            f"optimum_electrical_maximal_runs_{variable.replace('/', '_')}.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

        # Plot the electrical fitness with all points on the Pareto front.
        plt.figure(figsize=(48 / 5, 32 / 5))

        sns.scatterplot(
            runs_data[
                (runs_data["electrical_fitness"] > 0)
                & (runs_data["thermal_fitness"] > 0)
            ],
            x=variable,
            y="normalised_electrical_fitness",
            # color="grey",
            hue="run_number",
            palette=un_color_palette,
            marker="h",
            s=200,
            alpha=0.05,
            linewidth=0,
            legend=False,
        )
        sns.scatterplot(
            pareto_front_frame,
            x=variable,
            y="normalised_electrical_fitness",
            hue="run_number",
            palette=[
                color
                for index, color in enumerate(un_color_palette)
                if index in set(pareto_front_frame["run_number"])
            ],
            s=200,
            marker="h",
        )

        plt.xlabel(variable_labels[variable])
        plt.ylabel(
            r"Normalised electrical energy produced ($f_\mathrm{el}$) / kWh$_\mathrm{el}$/kWh$_\mathrm{in}$"
        )
        plt.ylim(0, 1)

        handles, plotted_labels = (axis := plt.gca()).get_legend_handles_labels()
        plt.legend(handles, [labels[int(label)] for label in plotted_labels])

        plt.savefig(
            f"optimum_electrical_on_pareto_{variable.replace('/', '_')}.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

        # Plot the points on the Pareto front and which are optimal.
        plt.figure(figsize=(48 / 5, 32 / 5))

        sns.scatterplot(
            runs_data[
                (runs_data["electrical_fitness"] > 0)
                & (runs_data["thermal_fitness"] > 0)
            ],
            x=variable,
            y="normalised_electrical_fitness",
            # color="grey",
            hue="run_number",
            palette=un_color_palette,
            marker="h",
            s=200,
            alpha=0.05,
            linewidth=0,
            legend=False,
        )
        sns.scatterplot(
            pareto_front_frame,
            x=variable,
            y="normalised_electrical_fitness",
            hue="run_number",
            palette=[
                color
                for index, color in enumerate(un_color_palette)
                if index in set(pareto_front_frame["run_number"])
            ],
            s=200,
            marker="h",
        )
        sns.scatterplot(
            maximal_runs,
            x=variable,
            y="normalised_electrical_fitness",
            hue="run_number",
            palette=un_color_palette,
            s=200,
            alpha=1.0,
            marker="h",
        )

        plt.xlabel(variable_labels[variable])
        plt.ylabel(
            r"Normalised electrical energy produced ($f_\mathrm{el}$) / kWh$_\mathrm{el}$/kWh$_\mathrm{in}$"
        )
        plt.ylim(0, 1)

        handles, plotted_labels = (axis := plt.gca()).get_legend_handles_labels()
        plt.legend(handles[-7:], list(labels.values())[-7:])

        plt.savefig(
            f"optimum_electrical_on_pareto_with_optimal_{variable.replace('/', '_')}.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

        # Plot the thermal fitness with only maximal values.
        plt.figure(figsize=(48 / 5, 32 / 5))

        sns.scatterplot(
            runs_data[
                (runs_data["electrical_fitness"] > 0)
                & (runs_data["thermal_fitness"] > 0)
            ],
            x=variable,
            y="normalised_thermal_fitness",
            # color="grey",
            hue="run_number",
            palette=un_color_palette,
            marker="h",
            s=200,
            alpha=0.05,
            linewidth=0,
            legend=False,
        )
        sns.scatterplot(
            maximal_runs,
            x=variable,
            y="normalised_thermal_fitness",
            hue="run_number",
            palette=un_color_palette,
            s=200,
            alpha=1.0,
            marker="h",
        )

        plt.xlabel(variable_labels[variable])
        plt.ylabel(
            r"Normalised thermal energy produced ($f_\mathrm{th}$) / kWh$_\mathrm{th}$/kWh$_\mathrm{in}$"
        )
        plt.ylim(0, 1)

        handles, plotted_labels = (axis := plt.gca()).get_legend_handles_labels()
        plt.legend(handles, [labels[int(label)] for label in plotted_labels])

        plt.savefig(
            f"optimum_thermal_maximal_runs_{variable.replace('/', '_')}.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

        # Plot the thermal fitness with all points on the Pareto front.
        plt.figure(figsize=(48 / 5, 32 / 5))

        sns.scatterplot(
            runs_data[
                (runs_data["electrical_fitness"] > 0)
                & (runs_data["thermal_fitness"] > 0)
            ],
            x=variable,
            y="normalised_thermal_fitness",
            # color="grey",
            hue="run_number",
            palette=un_color_palette,
            marker="h",
            s=200,
            alpha=0.05,
            linewidth=0,
            legend=False,
        )
        sns.scatterplot(
            pareto_front_frame,
            x=variable,
            y="normalised_thermal_fitness",
            hue="run_number",
            palette=[
                color
                for index, color in enumerate(un_color_palette)
                if index in set(pareto_front_frame["run_number"])
            ],
            s=200,
            marker="h",
        )

        plt.xlabel(variable_labels[variable])
        plt.ylabel(
            r"Normalised thermal energy produced ($f_\mathrm{th}$) / kWh$_\mathrm{th}$/kWh$_\mathrm{in}$"
        )
        plt.ylim(0, 1)

        handles, plotted_labels = (axis := plt.gca()).get_legend_handles_labels()
        plt.legend(handles, [labels[int(label)] for label in plotted_labels])

        plt.savefig(
            f"optimum_thermal_on_pareto_{variable.replace('/', '_')}.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

        # Plot the thermal fitness with all points on the Pareto front.
        plt.figure(figsize=(48 / 5, 32 / 5))

        sns.scatterplot(
            runs_data[
                (runs_data["electrical_fitness"] > 0)
                & (runs_data["thermal_fitness"] > 0)
            ],
            x=variable,
            y="normalised_thermal_fitness",
            # color="grey",
            hue="run_number",
            palette=un_color_palette,
            marker="h",
            s=200,
            alpha=0.05,
            linewidth=0,
            legend=False,
        )
        sns.scatterplot(
            pareto_front_frame,
            x=variable,
            y="normalised_thermal_fitness",
            hue="run_number",
            palette=[
                color
                for index, color in enumerate(un_color_palette)
                if index in set(pareto_front_frame["run_number"])
            ],
            s=200,
            marker="h",
        )
        sns.scatterplot(
            maximal_runs,
            x=variable,
            y="normalised_thermal_fitness",
            hue="run_number",
            palette=un_color_palette,
            s=200,
            alpha=1.0,
            marker="h",
        )

        plt.xlabel(variable_labels[variable])
        plt.ylabel(
            r"Normalised thermal energy produced ($f_\mathrm{th}$) / kWh$_\mathrm{th}$/kWh$_\mathrm{in}$"
        )
        plt.ylim(0, 1)

        handles, plotted_labels = (axis := plt.gca()).get_legend_handles_labels()
        plt.legend(handles[-7:], list(labels.values())[-7:])

        plt.savefig(
            f"optimum_thermal_on_pareto_with_optimal_{variable.replace('/', '_')}.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

    plt.show()

    import pdb

    pdb.set_trace()

    return maximal_runs, runs_data


def main(unparsed_args: list[Any]) -> None:
    """
    The main function for the module.

    :param: unparsed_args
        The unparsed command-line arguments.

    """

    # Parse the command-line arguments.
    parsed_args = _parse_args(unparsed_args)
    (base_collector_filepath, base_model_input_filepaths) = _validate_args(parsed_args)

    # Determine the date and time if not provided.
    date_and_time = DateAndTime(parsed_args.date, parsed_args.time)

    # Open the configuration files necessary for the run.
    (
        collector_model_assessors,
        optimisation_parameters,
        weather_data_sample,
        weather_data_full,
    ) = _parse_files(
        base_collector_filepath,
        base_model_input_filepaths,
        date_and_time,
        parsed_args.location,
        resample=parsed_args.resample,
        weather_sample_filename=parsed_args.weather_sample_filename,
        weather_sample_size=parsed_args.weather_sample_size,
    )

    # Parse the weightings information
    electrical_weightings = {
        index: collector_model_assessors[
            index
        ].weighting_calculator.electrical_weighting
        for index in range(len(collector_model_assessors))
    }
    thermal_weightings = {
        index: collector_model_assessors[index].weighting_calculator.thermal_weighting
        for index in range(len(collector_model_assessors))
    }

    if parsed_args.plotting_only:
        # Determine the number of steady-state runs that took place.
        with open(base_model_input_filepaths[0], "r") as steady_state_file:
            num_repeats = len(yaml.safe_load(steady_state_file))
            # num_repeats = 1

        maximal_runs, runs_data = plot_pareto_front(
            date_and_time,
            optimisation_parameters,
            weather_data_sample,
            electrical_weightings=electrical_weightings,
            num_repeats=num_repeats,
            thermal_weightings=thermal_weightings,
        )

        return

    # Run the Bayesian optimiser threads
    # bayesian_assessor_0 = BayesianPVTModelOptimiserSeries(
    #     date_and_time,
    #     optimisation_parameters,
    #     collector_model_assessors[0],
    #     (collector_model_index_to_results_map := {}),
    #     weather_data_sample[WeatherDataHeader.SOLAR_IRRADIANCE.value],
    #     weather_data_sample[WeatherDataHeader.AMBIENT_TEMPERATURE.value],
    #     weather_data_sample[WeatherDataHeader.WIND_SPEED.value],
    #     initial_points=(_initial_points := 10),
    #     num_iterations=(_num_iterations := 4),
    #     run_id=0,
    # )
    # bayesian_assessor_0.run()

    # bayesian_assessor_1 = BayesianPVTModelOptimiserSeries(
    #     date_and_time,
    #     optimisation_parameters,
    #     collector_model_assessors[1],
    #     (collector_model_index_to_results_map := {}),
    #     weather_data_sample[WeatherDataHeader.SOLAR_IRRADIANCE.value],
    #     weather_data_sample[WeatherDataHeader.AMBIENT_TEMPERATURE.value],
    #     weather_data_sample[WeatherDataHeader.WIND_SPEED.value],
    #     initial_points=_initial_points,
    #     num_iterations=_num_iterations,
    #     run_id=1,
    # )
    # bayesian_assessor_1.run()

    # bayesian_assessor_2 = BayesianPVTModelOptimiserSeries(
    #     date_and_time,
    #     optimisation_parameters,
    #     collector_model_assessors[2],
    #     (collector_model_index_to_results_map := {}),
    #     weather_data_sample[WeatherDataHeader.SOLAR_IRRADIANCE.value],
    #     weather_data_sample[WeatherDataHeader.AMBIENT_TEMPERATURE.value],
    #     weather_data_sample[WeatherDataHeader.WIND_SPEED.value],
    #     initial_points=_initial_points,
    #     num_iterations=_num_iterations,
    #     run_id=2,
    # )
    # bayesian_assessor_2.run()

    # bayesian_assessor_3 = BayesianPVTModelOptimiserSeries(
    #     date_and_time,
    #     optimisation_parameters,
    #     collector_model_assessors[3],
    #     (collector_model_index_to_results_map := {}),
    #     weather_data_sample[WeatherDataHeader.SOLAR_IRRADIANCE.value],
    #     weather_data_sample[WeatherDataHeader.AMBIENT_TEMPERATURE.value],
    #     weather_data_sample[WeatherDataHeader.WIND_SPEED.value],
    #     initial_points=_initial_points,
    #     num_iterations=_num_iterations,
    #     run_id=3,
    # )
    # bayesian_assessor_3.run()

    # Setup the Bayesian optimiser threads.
    bayesian_assessors: list[BayesianPVTModelOptimiserThread] = []
    collector_model_index_to_results_map: dict[str, dict[str, float] | float] = {}
    for index, collector_model_assessor in enumerate(collector_model_assessors):
        if collector_model_assessor.collector_type == CollectorType.PVT:
            bayesian_assessors.append(
                bayesian_assessor := BayesianPVTModelOptimiserThread(
                    date_and_time,
                    optimisation_parameters,
                    collector_model_assessor,
                    collector_model_index_to_results_map,
                    weather_data_sample[WeatherDataHeader.SOLAR_IRRADIANCE.value],
                    weather_data_sample[WeatherDataHeader.AMBIENT_TEMPERATURE.value],
                    weather_data_sample[WeatherDataHeader.WIND_SPEED.value],
                    initial_points=parsed_args.initial_points,
                    num_iterations=parsed_args.num_iterations,
                    run_id=index,
                )
            )
            bayesian_assessor.start()

    for bayesian_assessor in bayesian_assessors:
        bayesian_assessor.join()

    # Determine the number of steady-state runs that took place.
    with open(base_model_input_filepaths[0], "r") as steady_state_file:
        num_repeats = len(yaml.safe_load(steady_state_file))

    plot_pareto_front(
        date_and_time,
        optimisation_parameters,
        weather_data_sample,
        electrical_weightings=electrical_weightings,
        thermal_weightings=thermal_weightings,
    )

    import pdb

    pdb.set_trace()

    # Construct a Bayseian optimiser based on the inputs.

    # Optimise based on the inputs.

    # Save the results and display the outputs.


if __name__ == "__main__":
    main(sys.argv[1:])
