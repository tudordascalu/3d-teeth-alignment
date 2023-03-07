"""
Define the parameters that the scripts can use.
"""
import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="The script handles pre-processing of the data.")
    parser.add_argument("-j", "--jaw", type=str, help="The jaw to focus on.", default="upper",
                        choices=["lower", "upper"])
    parser.add_argument("-d", "--dir", type=str, help="The type of data to act on.", default="final",
                        choices=["processed", "final"])
    parser.add_argument("-s", "--max_swaps", type=int, help="Maximum number of swaps.", default=2)
    parser.add_argument("-t", "--teeth", type=int, help="Number of teeth.", default=17)
    parser.add_argument("-n", "--n_samples", type=int, help="The number of samples created for each patient.",
                        default=10)
    parser.add_argument("-pd", "--p_dummy", type=float, help="The probability of injecting a dummy tooth.",
                        default=.5)
    parser.add_argument("-pm", "--p_missing", type=float, help="The probability of removing a tooth.",
                        default=.5)
    parser.add_argument("-ps", "--p_swap", type=float, help="The probability of swapping a pair of teeth.",
                        default=.5)
    parser.add_argument("-mm", "--max_missing", type=int, help="The probability of removing a tooth.",
                        default=4)
    parser.add_argument("-md", "--max_distance", type=int,
                        help="The maximum distance between predicted and ground truth centroids.",
                        default=3)
    return parser
