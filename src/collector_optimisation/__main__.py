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
import os
import sys

from dataclasses import dataclass
from typing import Any

import pandas as pd
import yaml


# INPUT_FILES_DIRECTORY:
#   The name of the input-files directory.
INPUT_FILES_DIRECTORY: str = "input_files"

# LOCATIONS_FILENAME:
#   The name of the file containing the locations information.
LOCATIONS_FILENAME: str = "locations.yaml"

# SOLAR_FILENAME:
#   The name of the solar filename.
SOLAR_FILENAME: str = "ninja_pv_{lat:.4f}_{lon:.4f}_uncorrected.csv"

# WEATHER_DIRECTORY:
#   The name of the directory containing the weather information.
WEATHER_DIRECTORY: str = "weather_data"

# WIND_FILENAME:
#   The name of the wind filename.
WIND_FILENAME: str = "ninja_wind_{lat:.4f}_{lon:.4f}_uncorrected.csv"


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

    return parser.parse_args(args)


def _parse_files(location_name: str) -> None:
    """
    Parse the input files needed to run the model.

    :param: location_name
        The name of the location to consider.

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

    import pdb

    pdb.set_trace()

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
        solar_data = pd.read_csv(solar_file)

    with open(
        os.path.join(
            INPUT_FILES_DIRECTORY,
            WEATHER_DIRECTORY,
            WIND_FILENAME.format(lat=location.lat, lon=location.lon),
        ),
        "r",
        encoding="UTF-8",
    ) as wind_file:
        wind_data = pd.read_csv(wind_file)

    # Combine the two dataframes
    combined_weather_data = solar_data
    combined_weather_data["wind_speed"] = wind_data["wind_speed"]

    # Process the weather data to find unique values.


def main(unparsed_args: list[Any]) -> None:
    """
    The main function for the module.

    :param: unparsed_args
        The unparsed command-line arguments.

    """

    # Parse the command-line arguments.
    parsed_args = _parse_args(unparsed_args)

    # Open the configuration files necessary for the run.
    _parse_files(parsed_args.location)

    # Construct a Bayseian optimiser based on the inputs.

    # Optimise based on the inputs.

    # Save the results and display the outputs.

    pass


if __name__ == "__main__":
    main(sys.argv[1:])
