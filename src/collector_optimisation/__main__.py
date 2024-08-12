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
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from .__utils__ import INPUT_FILES_DIRECTORY, WeatherDataHeader
from .model_wrapper import CollectorModelAssessor, CollectorType, WeightingCalculator


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

# WEATHER_DIRECTORY:
#   The name of the directory containing the weather information.
WEATHER_DIRECTORY: str = "weather_data"

# WIND_FILENAME:
#   The name of the wind filename.
WIND_FILENAME: str = "ninja_wind_{lat:.4f}_{lon:.4f}_corrected.csv"


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

    return parser.parse_args(args)


def _parse_files(
    base_collector_filename: str,
    base_model_input_files: list[str],
    location_name: str,
    *,
    sample_type: SampleType = SampleType.DENSITY,
    weather_sample_size: int = 40,
) -> tuple[list[CollectorModelAssessor], pd.DataFrame, pd.DataFrame]:
    """
    Parse the input files needed to run the model.

    :param: base_collector_filename
        The base collector filename on which adjustments are made.

    :param: base_model_input_files
        A `list` of filenames which go into the underlying model.

    :param: location_name
        The name of the location to consider.

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

    density_based_weather_sample = _density_based_sampling(
        modelling_weather_array, weather_sample_size, bandwidth=0.4
    )
    grid_based_weather_sample = _weighted_grid_sampling(
        modelling_weather_array, [50, 5, 1], weather_sample_size
    )

    ###################################################################
    # Code for plotting and visualising the weather-data distribution #
    ###################################################################

    # sns.set_context("notebook")
    # sns.set_style("ticks")

    # sns.set_palette(
    #     [
    #         "#2CBCE0",
    #         "#0D699F",
    #     ]
    # )

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
    #     x=density_based_weather_sample[:, 0],
    #     y=density_based_weather_sample[:, 1],
    #     s=100,
    #     marker="H",
    #     alpha=0.8,
    #     linewidth=0,
    #     color="C1",
    # )
    # plt.savefig("scatter_plot_density_weather.pdf", bbox_inches="tight", pad_inches=0)

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

    # sns.jointplot(
    #     modelling_weather_data,
    #     x="irradiance_total",
    #     y="temperature",
    #     linewidth=0,
    #     height=32 / 5,
    #     ratio=4,
    #     marginal_kws={"bins": 20},
    #     kind="hex",
    # )
    # ax = plt.gca()
    # ax.set_xlabel("Irradiance / W/m$^2$")
    # ax.set_ylabel("Temperature / $^\circ$C")
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # plt.savefig("hex_plot_weather_data.pdf", bbox_inches="tight", pad_inches=0)

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

    # Return the sampled weather data alone with the model assessors.
    weather_sample = pd.DataFrame(
        density_based_weather_sample
        if sample_type == SampleType.DENSITY
        else grid_based_weather_sample
    )
    weather_sample.columns = pd.Index(
        (
            column_headers := [
                WeatherDataHeader.SOLAR_IRRADIANCE.value,
                WeatherDataHeader.AMBIENT_TEMPERATURE.value,
                WeatherDataHeader.WIND_SPEED.value,
            ]
        )
    )

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


def main(unparsed_args: list[Any]) -> None:
    """
    The main function for the module.

    :param: unparsed_args
        The unparsed command-line arguments.

    """

    # Parse the command-line arguments.
    parsed_args = _parse_args(unparsed_args)
    (base_collector_filepath, base_model_input_filepaths) = _validate_args(parsed_args)

    # Open the configuration files necessary for the run.
    (
        collector_model_assessors,
        optimisation_parameters,
        weather_data_sample,
        weather_data_full,
    ) = _parse_files(
        base_collector_filepath,
        base_model_input_filepaths,
        parsed_args.location,
        weather_sample_size=parsed_args.weather_sample_size,
    )

    collector_model_assessors[0].fitness_function(
        0.5,
        0,
        weather_data_sample[WeatherDataHeader.SOLAR_IRRADIANCE.value],
        weather_data_sample[WeatherDataHeader.AMBIENT_TEMPERATURE.value],
        weather_data_sample[WeatherDataHeader.WIND_SPEED.value],
    )

    import pdb

    pdb.set_trace()

    # Construct a Bayseian optimiser based on the inputs.

    # Optimise based on the inputs.

    # Save the results and display the outputs.

    pass


if __name__ == "__main__":
    main(sys.argv[1:])
