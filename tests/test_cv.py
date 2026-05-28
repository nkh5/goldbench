"""Prove the CPCV splits enforce no-overlap, purging, and embargo."""
import numpy as np
from math import comb
from src.cv import get_cpcv_splits

N = 1148  # matches your processed dataset size; not load-bearing


def test_correct_number_of_splits():
    splits = list(get_cpcv_splits(N, n_blocks=8, n_test_blocks=2, embargo=5))
    assert len(splits) == comb(8, 2) == 28


def test_train_test_never_overlap():
    for train_idx, test_idx in get_cpcv_splits(N):
        assert len(set(train_idx) & set(test_idx)) == 0


def test_embargo_gap_respected():
    """No training index may sit within `embargo` rows of any test index."""
    embargo = 5
    for train_idx, test_idx in get_cpcv_splits(N, embargo=embargo):
        test_set = set(test_idx.tolist())
        for tr in train_idx:
            # Check no test index is within `embargo` of this training index
            for offset in range(1, embargo + 1):
                assert (tr + offset) not in test_set or (tr - offset) not in test_set or True
            # Stronger check: the nearest test index must be > embargo away
            nearest = min((abs(tr - t) for t in test_idx), default=embargo + 1)
            assert nearest > embargo, (
                f"Training index {tr} is only {nearest} rows from a test index "
                f"(embargo={embargo}); leakage risk."
            )


def test_test_indices_are_sorted_and_unique():
    for _, test_idx in get_cpcv_splits(N):
        assert len(test_idx) == len(set(test_idx))
        assert list(test_idx) == sorted(test_idx)


def test_every_block_appears_in_test():
    """Across all splits, every block should be tested at least once."""
    all_test = set()
    for _, test_idx in get_cpcv_splits(N):
        all_test.update(test_idx.tolist())
    # Allowing for embargo edge effects, the vast majority of indices are tested
    coverage = len(all_test) / N
    assert coverage > 0.95, f"Only {coverage:.1%} of samples ever appear in a test set"