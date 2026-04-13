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
import math

__all__ = (
    "AMBIENT",
    "COLLECTOR_INPUT_TEMPERATURE",
    "DateAndTime",
    "HALF_WAY",
    "INPUT_FILES_DIRECTORY",
    "parasitic_power_loss",
    "parasitic_pressure_loss",
    "WeatherDataHeader",
)


# AMBIENT:
#   Keyword used to parse when a stead-state run specifies that the ambient temperature
# should be used.
AMBIENT: str = "AMBIENT"

# COLLECTOR_INPUT_TEMPERATURE:
#   Keyword used for parsing the collector input-temperature information.
COLLECTOR_INPUT_TEMPERATURE: str = "collector_input_temperature"

# HALF_WAY:
#   Keyword used to parse when a stead-state run specifies that a temperature half-way
# (i.e., the mean) between the ambient temperature and the max value should be used.
HALF_WAY: str = "HALF_WAY"

# INPUT_FILES_DIRECTORY:
#   The name of the input-files directory.
INPUT_FILES_DIRECTORY: str = "input_files"

# ZERO_CELCIUS_OFFSET:
#   The temperature of absolute zero in Kelvin, used for converting Celcius to Kelvin
# and vice-a-versa.
ZERO_CELCIUS_OFFSET: float = 273.15


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


def darcy_friction_factor(reynolds_number: float) -> float:
    """
    Computes and returns the Darcy friction factor.

    The formula used for the friction factor is obtained from:
    Inglesais-Manríquez, E., Brottier, L. & Bennacer, R.
    Pressure Drop in Parallel Flow Flat-Plate PVT Collectors
    in International Solar Energy Society (ISES) Conference Proceedings (2016).

    :param: reynolds_number:
        The Reynolds number of the fluid.

    :returns:
        The Darcy friction factor.

    """

    if reynolds_number <= 2300:
        return 64 / reynolds_number

    return 0.3164 * (reynolds_number**0.25)


def density_of_water(fluid_temperature: float) -> float:
    """
    The density of water varies as a function of temperature.

    The formula for the density is obtained from:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4909168/

    :param fluid_temperature:
        The temperature of the fluid, measured in Kelvin.

    :return:
        The density of water, measured in kilograms per meter cubed.

    """

    return (
        999.85308
        + 6.32693 * (10 ** (-2)) * (fluid_temperature - ZERO_CELCIUS_OFFSET)
        - 8.523892 * (10 ** (-3)) * (fluid_temperature - ZERO_CELCIUS_OFFSET) ** 2
        + 6.943249 * (10 ** (-5)) * (fluid_temperature - ZERO_CELCIUS_OFFSET) ** 3
        - 3.82126 * (10 ** (-7)) * (fluid_temperature - ZERO_CELCIUS_OFFSET) ** 4
    )


def dynamic_viscosity_of_water(fluid_temperature: float) -> float:
    """
    The dynamic viscosity of water varies as a function of temperature.

    The formula comes from the Vogel-Fulcher-Tammann equation via Wiki:
    https://en.wikipedia.org/wiki/Viscosity#Water

    :param fluid_temperature:
        The temperature of the fluid being modelled, measured in Kelvin.

    :return:
        The dynamic viscosity of water, measured in kilograms per meter second.

    """

    return 0.00002939 * math.exp(507.88 / (fluid_temperature - 149.3))


def parasitic_pressure_loss(
    characteristic_diameter: float,
    characteristic_length: float,
    fluid_density: float | None,
    fluid_temperature: float,
    fluid_velocity: float,
) -> float:
    """
    Compute the parasitic pressure loss, in W, of the pump.

    The electrical output of the PV-T collectors will be reduced by the reqyurements of
    the pump needed to move the HTF through the collectors. For an incompressible fluid
    with a negligible or neglected height difference, this loss comes purely from
    friction:
        Δp_loss = f * ρv^2 L / 2D,
    where:
        f --- is the Darcy--Weisbach friction coefficient,
        ρ --- "rho" is the density of the fluid,
        v --- the velocity of the fluid,
        L --- a charactieristic length scale,
    and D --- the diameter of the fluid.

    :param: characteristic_diameter:
        The characteristic diameter of the pipe.

    :param: characteristic_length:
        The characteristic length of the collector.

    :param: fluid_desnsity:
        The density of the fluid

    :param: fluid_temperature:
        The temperature of the fluid, in Celcius.

    :param: fluid_velocity:
        The velocity of the fluid in m/s.

    """

    if isinstance(fluid_temperature, list) and isinstance(fluid_velocity, list):
        if len(fluid_temperature) != len(fluid_velocity):
            raise Exception(
                "Fluid temperature and fluid velocity must have the same length."
            )

        if fluid_density is None:
            fluid_density = [
                density_of_water(entry + ZERO_CELCIUS_OFFSET)
                for entry in fluid_temperature
            ]

        return [
            (
                fluid_density[index]
                * darcy_friction_factor(
                    reynolds_number(
                        fluid_density[index],
                        dynamic_viscosity_of_water(
                            fluid_temperature[index] + ZERO_CELCIUS_OFFSET,
                        ),
                        fluid_velocity[index],
                        characteristic_diameter,
                    )
                )
                * characteristic_length
                * (fluid_velocity[index] ** 2)
                / (2 * characteristic_diameter)
            )
            for index, _ in enumerate(fluid_temperature)
        ]

    if isinstance(fluid_temperature, float) and isinstance(fluid_velocity, float):
        if fluid_density is None:
            fluid_density = density_of_water(fluid_temperature + ZERO_CELCIUS_OFFSET)

        return (
            fluid_density
            * darcy_friction_factor(
                reynolds_number(
                    fluid_density,
                    dynamic_viscosity_of_water(
                        fluid_temperature + ZERO_CELCIUS_OFFSET,
                    ),
                    fluid_velocity,
                    characteristic_diameter,
                )
            )
            * characteristic_length
            * (fluid_velocity**2)
            / (2 * characteristic_diameter)
        )

    raise Exception("Fluid temperature and fluid velocity must be of the same type.")


def parasitic_power_loss(
    characteristic_diameter: float,
    characteristic_length: float,
    fluid_density: float | None,
    fluid_temperature: float | list[float],
    fluid_velocity: float | list[float],
    mass_flow_rate: float,
) -> float | list[float]:
    """
    Compute the parasitic power loss for the collector.

    :param: characteristic_diameter:
        The characteristic diameter of the pipe.

    :param: characteristic_length:
        The characteristic length of the collector.

    :param: fluid_desnsity:
        The density of the fluid

    :param: fluid_temperature:
        The temperature of the fluid, in Celcius.

    :param: fluid_velocity:
        The velocity of the fluid in m/s.

    :param: mass_flow_rate:
        The mass flow rate, in kg/s, of the fluid.

    """

    if isinstance(fluid_temperature, list) and isinstance(fluid_velocity, list):
        if len(fluid_temperature) != len(fluid_velocity):
            raise Exception(
                "Fluid temperature and fluid velocity must have the same length."
            )

        if fluid_density is None:
            fluid_density = [
                density_of_water(entry + ZERO_CELCIUS_OFFSET)
                for entry in fluid_temperature
            ]

        return [
            (
                mass_flow_rate
                * parasitic_pressure_loss(
                    characteristic_diameter,
                    characteristic_length,
                    fluid_density[index],
                    fluid_temperature[index],
                    fluid_velocity[index],
                )
                / fluid_density[index]
            )
            for index, _ in enumerate(fluid_temperature)
        ]

    if isinstance(fluid_temperature, float) and isinstance(fluid_velocity, float):
        if fluid_density is None:
            fluid_density = density_of_water(fluid_temperature + ZERO_CELCIUS_OFFSET)

        return (
            mass_flow_rate
            * parasitic_pressure_loss(
                characteristic_diameter,
                characteristic_length,
                fluid_density,
                fluid_temperature,
                fluid_velocity,
            )
            / fluid_density
        )

    raise Exception("Fluid temperature and fluid velocity must be of the same type.")


def reynolds_number(
    density: float, dynamic_viscosity: float, flow_speed: float, length_scale: float
) -> float:
    """
    Computes the Reynolds number of the flow.

    :param: density:
        The density of the fluid, measured in kilograms per meter cubed.

    :param: dynamic_viscosity:
        The dynamic viscosity, measured in kilograms per meter second.

    :param: flow_speed:
        The speed of the flow, measured in meters per second.

    :param: length_scale:
        A characteristic length scale over which Physics in the fluid is occurring.

    :return:
        The dimensionless Reynolds number.

    """

    return (
        density  # [kg/m^3]
        * flow_speed  # [m/s]
        * length_scale  # [m]
        / dynamic_viscosity  # [kg/m*s]
    )
