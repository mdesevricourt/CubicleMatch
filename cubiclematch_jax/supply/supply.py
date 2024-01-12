import logging
import pprint

pp = pprint.PrettyPrinter(indent=4)
import itertools
from math import comb
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def generate_bundles(
    max_num_half_days: int, tot_num_half_days: int, num_cubicles: int, verbose=True
):
    """Generate all possible supply vectors. The supply vectors are of length `tot_num_half_days * num_cubicles` and
    have the following structure: The first `tot_num_half_days` entries correspond to the first cubicle, the next
    `tot_num_half_days` entries correspond to the second cubicle, and so on. The entries are either 0 or 1.

    Args:
        max_num_half_days (int): The maximum number of half-days per agent.
        tot_num_half_days (int): The total number of half-days.
        num_cubicles (int): The number of cubicles.
        verbose (bool, optional): Whether to print progress. Defaults to True.

    Returns:
        np.ndarray: The supply vectors."""
    vector_length = tot_num_half_days * num_cubicles
    possible_cubicle = np.arange(num_cubicles + 1)
    possible_half_days = np.arange(tot_num_half_days)

    n_possible_combinations = (num_cubicles + 1) ** max_num_half_days * comb(
        tot_num_half_days, max_num_half_days
    )
    valid_bundles = np.zeros((n_possible_combinations, vector_length), dtype=bool)
    print(valid_bundles.shape)
    num_selection_half_days = comb(tot_num_half_days, max_num_half_days)

    for n, selected_half_days in enumerate(
        itertools.combinations(possible_half_days, max_num_half_days)
    ):
        if verbose:
            logging.info(f"Generating vectors for {selected_half_days}")
            logging.info(f"Progress: {n}/{num_selection_half_days}")
        for i, vector in enumerate(
            itertools.product(possible_cubicle, repeat=max_num_half_days)
        ):
            num_iter = i + n * (num_cubicles + 1) ** max_num_half_days

            for j, cubicle in enumerate(vector):
                half_day = selected_half_days[j]
                if cubicle < num_cubicles:
                    position = half_day + cubicle * tot_num_half_days
                    valid_bundles[num_iter, position] = 1

    # only keep one vector of zeros

    valid_bundles = valid_bundles[~np.all(valid_bundles == 0, axis=1)]

    # add one vector of only zeros backs

    valid_bundles = np.vstack((valid_bundles, np.zeros(vector_length, dtype=bool)))

    return valid_bundles.astype(np.float32)


def save_vectors(vector, filename):
    np.save(filename, vector)


def generate_and_save_bundles(
    max_num_half_days: int,
    total_num_half_days: int,
    total_num_cubicles: int,
    data_path: Path = Path(""),
):
    logging.info(
        f"Generating vectors for {total_num_half_days} half-days and {total_num_cubicles} cubicles"
    )
    filename = (
        f"vectors_{max_num_half_days}_{total_num_half_days}_{total_num_cubicles}.npy"
    )
    file_path = data_path / filename
    if file_path.exists():
        logging.info(f"File {filename} already exists")
        return

    vectors = generate_bundles(
        max_num_half_days=max_num_half_days,
        tot_num_half_days=total_num_half_days,
        num_cubicles=total_num_cubicles,
    )

    save_vectors(vectors, file_path)
    logging.info(f"Saved vectors to {filename}")


if __name__ == "__main__":
    generate_and_save_bundles(
        max_num_half_days=6, total_num_half_days=10, total_num_cubicles=6
    )
