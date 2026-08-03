"""Tests for the purged + embargoed walk-forward splitter."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from overfitcheck.crossval import PurgedWalkForwardSplitter, default_walk_forward_splitter


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        PurgedWalkForwardSplitter(0, 2, 2, 0, 1)
    with pytest.raises(ValueError):
        PurgedWalkForwardSplitter(4, 0, 2, 0, 1)
    with pytest.raises(ValueError):
        PurgedWalkForwardSplitter(4, 2, 2, -1, 1)
    with pytest.raises(ValueError):
        PurgedWalkForwardSplitter(4, 2, 2, 0, 0)


def test_single_fold_purge_no_embargo() -> None:
    spl = PurgedWalkForwardSplitter(
        train_size=4, valid_size=2, test_size=2, embargo_size=0, label_horizon=1
    )
    folds = spl.split(8)
    assert len(folds) == 1
    train, valid, test = folds[0]
    # raw train [0,1,2,3]; purge keeps i+1 < 4 -> {0,1,2}
    assert train.tolist() == [0, 1, 2]
    assert valid.tolist() == [4, 5]
    assert test.tolist() == [6, 7]


def test_embargo_removes_extra_bar() -> None:
    spl = PurgedWalkForwardSplitter(
        train_size=5, valid_size=2, test_size=2, embargo_size=2, label_horizon=1
    )
    train, _valid, _test = spl.split(9)[0]
    # purge (label_horizon=1) keeps {0,1,2,3}; embargo cutoff=5-2=3 keeps i<3 -> {0,1,2}
    assert train.tolist() == [0, 1, 2]


def test_positional_purge_with_long_horizon() -> None:
    spl = PurgedWalkForwardSplitter(
        train_size=10, valid_size=2, test_size=2, embargo_size=0, label_horizon=3
    )
    train, _valid, _test = spl.split(14)[0]
    # eval starts at 10; keep i where i+3 < 10 -> i <= 6
    assert train.tolist() == [0, 1, 2, 3, 4, 5, 6]


def test_walk_forward_steps_by_test_size_and_is_disjoint() -> None:
    spl = PurgedWalkForwardSplitter(
        train_size=6, valid_size=2, test_size=2, embargo_size=0, label_horizon=1
    )
    folds = spl.split(20)
    test_blocks = [test.tolist() for _t, _v, test in folds]
    # Test windows are contiguous, length test_size, and pairwise disjoint.
    flat = [i for block in test_blocks for i in block]
    assert len(flat) == len(set(flat))  # disjoint
    for block in test_blocks:
        assert len(block) == 2
        assert block[1] == block[0] + 1
    # Each fold steps forward by exactly test_size.
    starts = [block[0] for block in test_blocks]
    assert all(b - a == 2 for a, b in pairwise(starts))


def test_too_few_observations_raises() -> None:
    spl = PurgedWalkForwardSplitter(4, 2, 2, 0, 1)
    with pytest.raises(ValueError):
        spl.split(7)  # window is 8


def test_fully_purged_fold_is_skipped() -> None:
    # label_horizon >= train span -> every training obs purged -> no folds.
    spl = PurgedWalkForwardSplitter(
        train_size=2, valid_size=1, test_size=1, embargo_size=0, label_horizon=5
    )
    assert spl.split(4) == []


def test_accepts_datetimeindex_like_int() -> None:
    spl = PurgedWalkForwardSplitter(4, 2, 2, 0, 1)
    idx = pd.date_range("2020-01-01", periods=8, freq="B")
    by_index = spl.split(idx)
    by_int = spl.split(8)
    assert len(by_index) == len(by_int) == 1
    assert by_index[0][0].tolist() == by_int[0][0].tolist()


def test_default_splitter_produces_folds() -> None:
    spl = default_walk_forward_splitter(500)
    folds = spl.split(500)
    assert len(folds) >= 1
    for train, valid, test in folds:
        assert train.size > 0
        assert valid.size > 0
        assert test.size > 0


def test_window_property() -> None:
    spl = PurgedWalkForwardSplitter(10, 3, 3, 1, 1)
    assert spl.window == 16


def test_indices_are_integer_arrays() -> None:
    spl = PurgedWalkForwardSplitter(4, 2, 2, 0, 1)
    train, valid, test = spl.split(8)[0]
    for arr in (train, valid, test):
        assert isinstance(arr, np.ndarray)
        assert np.issubdtype(arr.dtype, np.integer)
