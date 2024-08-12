#!/usr/bin/python3.10
########################################################################################
# __utils__.py - Utility module for the collector-optimisation module.                 #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2024                                                      #
########################################################################################

"""
The utility module for the collector-optimisation software.

The utility module is responsible for providing common utility functions and helper code

"""

import enum

__all__ = (
    "INPUT_FILES_DIRECTORY",
    "WeatherDataHeader",
)


# INPUT_FILES_DIRECTORY:
#   The name of the input-files directory.
INPUT_FILES_DIRECTORY: str = "input_files"


class WeatherDataHeader(enum.Enum):
    """
    Used for categorising weather data.

    - AMBIENT_TEMPERATURE:
        Denotes the ambient temperature.

    - SOLAR_IRRADIANCE:
        Denotes the solar irradiance.

    - WIND_SPEED:
        Denotes the wind speed.

    """

    AMBIENT_TEMPERATURE: str = "ambient_temperature"
    SOLAR_IRRADIANCE: str = "irradiance"
    WIND_SPEED: str = "wind_speed"
