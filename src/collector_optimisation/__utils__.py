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

import datetime
import enum

__all__ = (
    "DateAndTime",
    "INPUT_FILES_DIRECTORY",
    "WeatherDataHeader",
)


# INPUT_FILES_DIRECTORY:
#   The name of the input-files directory.
INPUT_FILES_DIRECTORY: str = "input_files"


class DateAndTime:
    """
    Contains information about the date and time.

    .. attribute:: date
        The date, formatted as a string.

    .. attribute:: time
        The time, formatted as a string.

    """

    def __init__(self, date: str | None = None, time: str | None = None) -> None:
        """Instantiate based on the current date and time."""

        if date is None:
            date = (date_and_time := datetime.datetime.now()).date()
        else:
            try:
                date = datetime.datetime.strptime(date, "%d%m%y")
            except ValueError:
                try:
                    date = datetime.datetime.strptime(date, "%d/%m/%y")
                except ValueError:
                    try:
                        date = datetime.datetime.strptime(date, "%d_%m_%y")
                    except ValueError:
                        raise ValueError(
                            "Date, if specified, must be of DDMMYY, DD/MM/YY or "
                            "DD_MM_YY format."
                        )

        if time is None:
            time = date_and_time.time()
        else:
            try:
                time = datetime.datetime.strptime(time, "%H%M%S").time()
            except ValueError:
                try:
                    time = datetime.datetime.strptime(time, "%H:%M:%S")
                except ValueError:
                    try:
                        time = datetime.datetime.strptime(time, "%H_%M_%S")
                    except ValueError:
                        raise ValueError(
                            "Time, if specified, must be of HHMMSS, HH:MM:SS or "
                            "HH_MM_SS format."
                        )

        self.date = f"{date.day:02d}_{date.month:02d}_{date.year % 100:02d}"
        self.time = f"{time.hour:02d}_{time.minute:02d}_{time.second:02d}"

    def __repr__(self) -> str:
        """Return a nice-looking representation of the class."""

        return f"DateAndTime(date={self.date}, time={self.time})"


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
