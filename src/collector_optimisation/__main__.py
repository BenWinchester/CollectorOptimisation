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
import sys

from typing import Any

def _parse_args(args: list[Any]) -> argparse.Namespace:
    """
    Parses command-line arguments into a :class:`argparse.NameSpace`.

    :param: args
        The un-parsed command-line arguments.

    :returns: The parsed command-line arguments.

    """

    parser = argparse.ArgumentParser()

    return parser.parse_args(args)

def _parse_files() -> None:
    """
    Parse the input files needed to run the model.

    """

    pass

def main(unparsed_args: list[Any]) -> None:
    """
    The main function for the module.

    :param: unparsed_args
        The unparsed command-line arguments.

    """

    # Parse the command-line arguments.
    parsed_args = _parse_args(unparsed_args)

    # Open the configuration files necessary for the run.

    # Construct a Bayseian optimiser based on the inputs.

    # Optimise based on the inputs.

    # Save the results and display the outputs.

    pass

if __name__ == "__main__":
    main(sys.argv[1:])
