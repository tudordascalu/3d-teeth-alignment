"""
Define the parameters that the scripts can use.
"""
import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="The script handles pre-processing of the data.")
    parser.add_argument("-j", "--jaw", type=str, help="The jaw to focus on.", default="lower",
                        choices=["lower", "upper"])
    parser.add_argument("-d", "--dir", type=str, help="The type of data to act on.", default="processed",
                        choices=["processed", "final"])
    parser.add_argument("-s", "--swaps", type=int, help="Maximum number of swaps.", default=3)
    parser.add_argument("-t", "--teeth", type=int, help="Number of teeth.", default=17)
    return parser
