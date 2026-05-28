"""Combinatorial Purged Cross-Validation for time-series with overlapping labels.
"""

from itertools import combinations
import numpy as np
import pandas as pd


def get_cpcv_splits(
    n_samples: int,
    n_blocks: int = 8,
    n_test_blocks: int = 2,
    embargo: int = 5,
):
    """Yield (train_idx, test_idx) for every combination of test blocks.

    Parameters
    ----------
    n_samples : total number of rows in the (time-ordered) dataset
    n_blocks : number of contiguous blocks to partition the data into (N)
    n_test_blocks : how many blocks form the test set each split (k)
    embargo : number of rows to drop on each side of every test block
    Yields
    ------
    (train_idx, test_idx) : np.ndarray pairs of integer positions
    """
    indices = np.arange(n_samples)
    # Partition into n_blocks contiguous, near-equal blocks
    block_bounds = np.array_split(indices, n_blocks)

    for test_combo in combinations(range(n_blocks), n_test_blocks):
        test_idx = np.concatenate([block_bounds[b] for b in test_combo])
        test_idx_set = set(test_idx.tolist())

        # Build the embargoed exclusion zone around each test block
        embargoed = set()
        for b in test_combo:
            block = block_bounds[b]
            start, end = block[0], block[-1]
            lo = max(0, start - embargo)
            hi = min(n_samples - 1, end + embargo)
            embargoed.update(range(lo, hi + 1))

        # Training set = everything not in a test block and not embargoed
        train_idx = np.array(
            [i for i in indices if i not in test_idx_set and i not in embargoed]
        )
        yield train_idx, np.sort(test_idx)


def describe_splits(n_samples, n_blocks=8, n_test_blocks=2, embargo=5):
    """Print a human-readable summary of the split structure."""
    from math import comb
    splits = list(get_cpcv_splits(n_samples, n_blocks, n_test_blocks, embargo))
    n_paths = comb(n_blocks - 1, n_test_blocks - 1)
    print(f"Samples: {n_samples}")
    print(f"Blocks: {n_blocks}, test blocks per split: {n_test_blocks}, embargo: {embargo}")
    print(f"Total splits: {len(splits)} (expected C({n_blocks},{n_test_blocks})={comb(n_blocks, n_test_blocks)})")
    print(f"Reconstructable backtest paths: {n_paths}")
    avg_train = np.mean([len(tr) for tr, _ in splits])
    avg_test = np.mean([len(te) for _, te in splits])
    print(f"Avg train size: {avg_train:.0f}, avg test size: {avg_test:.0f}")


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/features.parquet")
    describe_splits(len(df))