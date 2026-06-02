"""Tools to help with submitting jobs to the cluster."""

from pathlib import Path

import numpy as np


def get_bigram(seed):
    """Return a random bigram of the form <adjective>_<noun>."""
    utils_dir = Path(__file__).parent
    adjectives = (utils_dir / "adjectives.txt").read_text().splitlines()
    nouns = (utils_dir / "nouns.txt").read_text().splitlines()

    rng = np.random.default_rng(seed)
    i_adj, i_noun = rng.choice(len(adjectives)), rng.choice(len(nouns))

    # Return the bigram with the words capitalized

    return adjectives[i_adj].capitalize() + nouns[i_noun].capitalize()
