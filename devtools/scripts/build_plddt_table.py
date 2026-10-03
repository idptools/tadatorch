"""
Build the table of predicted pLDDT scores for every possible 5-residue window.

One of the TADA features is the mean AlphaFold2 pLDDT score predicted for
each 5-residue feature window by alphaPredict. Running the alphaPredict
network was the slowest part of calculating features, but there are only 20^5
(3.2 million) possible windows, so this script runs alphaPredict on every one
and saves the results. tadatorch then looks the scores up in the table.

This script documents exactly how the table shipped with tadatorch was
generated, and only needs to be re-run if alphaPredict changes. It requires
alphaPredict, which is not a dependency of tadatorch itself. Each window is
predicted on its own (as TADA did), because alphaPredict gives very slightly
different scores when many windows are predicted together. It takes around 20
minutes of CPU time, split over the number of processes used.

Usage
-----
python build_plddt_table.py <path to output .npy> --processes 10
"""

import argparse
import multiprocessing

import alphaPredict
import numpy as np
import torch

from tadatorch.backend.constants import AMINO_ACIDS, FEATURE_WINDOW_SIZE
from tadatorch.backend.features import PLDDT_TABLE_UNIT, number_window

NUMBER_OF_WINDOWS: int = len(AMINO_ACIDS) ** FEATURE_WINDOW_SIZE

# Number of windows handed to a worker process at a time.
WINDOWS_PER_TASK: int = 20000

DEFAULT_PROCESSES: int = 4


def make_window(number: int) -> str:
    """
    Convert the position of an entry in the table into its 5-residue window.

    This is the reverse of ``tadatorch.backend.features.number_window()``.

    Parameters
    ----------
    number : int
        Position in the table, between 0 and 3,199,999.

    Returns
    -------
    str
        The 5-residue window with that position.
    """
    residues = []
    for _ in range(FEATURE_WINDOW_SIZE):
        number, residue_index = divmod(number, len(AMINO_ACIDS))
        residues.append(AMINO_ACIDS[residue_index])

    # the last residue of the window is the smallest digit, so comes out first
    window = "".join(reversed(residues))
    return window


def use_one_thread() -> None:
    """
    Stop PyTorch using more than one thread in a worker process.

    The worker processes already run in parallel, so letting each one start
    several threads would just make them compete for the same cores.
    """
    torch.set_num_threads(1)


def predict_plddt_sums(first_number: int) -> np.ndarray:
    """
    Predict the pLDDT scores of a run of consecutive windows.

    Parameters
    ----------
    first_number : int
        Position in the table of the first window to predict.

    Returns
    -------
    np.ndarray
        For each window, the sum of its five per-residue pLDDT scores in
        units of 0.0001, as an integer. alphaPredict rounds its scores to
        four decimal places, so no information is lost by storing integers.
    """
    last_number = min(first_number + WINDOWS_PER_TASK, NUMBER_OF_WINDOWS)

    sums = np.empty(last_number - first_number, dtype=np.int32)
    for index, number in enumerate(range(first_number, last_number)):
        window = make_window(number)
        assert number_window(window) == number, f"Numbering of {window} is not reversible."

        sums[index] = round(sum(alphaPredict.predict(window)) / PLDDT_TABLE_UNIT)

    return sums


def main() -> None:
    """Predict every window and save the table."""
    parser = argparse.ArgumentParser(
        description="Build the table of alphaPredict pLDDT scores for every 5-residue window."
    )
    parser.add_argument("output", help="Path the table (.npy) is written to.")
    parser.add_argument(
        "--processes",
        type=int,
        default=DEFAULT_PROCESSES,
        help=f"Number of processes to use. Default is {DEFAULT_PROCESSES}.",
    )
    arguments = parser.parse_args()

    first_numbers = range(0, NUMBER_OF_WINDOWS, WINDOWS_PER_TASK)
    with multiprocessing.Pool(arguments.processes, initializer=use_one_thread) as pool:
        table = np.concatenate(pool.map(predict_plddt_sums, first_numbers))

    assert table.shape == (NUMBER_OF_WINDOWS,), f"Expected {NUMBER_OF_WINDOWS} windows."

    np.save(arguments.output, table)
    print(f"Saved pLDDT scores for {NUMBER_OF_WINDOWS} windows to {arguments.output}")


if __name__ == "__main__":
    main()
