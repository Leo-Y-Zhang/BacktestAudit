"""Purged + embargoed walk-forward cross-validation.

Reference: Lopez de Prado (2018), *Advances in Financial Machine Learning*,
chapter 7 ("Cross-Validation in Finance"). The two leakage controls implemented
here are:

* **Purging** -- drop any training observation whose forward-label window
  overlaps the evaluation window. Crucially this is done *positionally* (by bar
  count), not by calendar days: on a business-day index a 5-*calendar*-day purge
  removes only ~3-4 trading bars and leaks the tail of the label window.
* **Embargo** -- additionally drop the last ``embargo_size`` training bars
  immediately before the evaluation window, to defend against look-ahead through
  lagged/serially-correlated features.

Validation and test windows are never purged or embargoed.
"""

from __future__ import annotations

import logging
from collections.abc import Sized

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

__all__ = ["PurgedWalkForwardSplitter", "default_walk_forward_splitter"]

IntArray = npt.NDArray[np.int_]
Fold = tuple[IntArray, IntArray, IntArray]


class PurgedWalkForwardSplitter:
    """Walk-forward splitter with positional purging and embargo.

    Windows slide forward in non-overlapping ``test_size`` steps, so each test
    block is disjoint from the others -- a true walk-forward, not a random
    k-fold. Indices returned are *positional* (integer offsets into the supplied
    timeline), so the splitter is agnostic to the calendar; only ordering counts.

    Parameters
    ----------
    train_size, valid_size, test_size:
        Window lengths in *bars* (observations). Each must be ``> 0``.
    embargo_size:
        Number of training bars to drop from the end of each train window
        (closest to the evaluation window). Must be ``>= 0``.
    label_horizon:
        Forward-label length in *bars* (not calendar days). Must be ``> 0``.
        An observation at position ``i`` is purged unless
        ``i + label_horizon < eval_start``.
    """

    def __init__(
        self,
        train_size: int,
        valid_size: int,
        test_size: int,
        embargo_size: int,
        label_horizon: int,
    ) -> None:
        if train_size <= 0 or valid_size <= 0 or test_size <= 0:
            raise ValueError("train_size, valid_size and test_size must all be > 0")
        if embargo_size < 0:
            raise ValueError("embargo_size must be >= 0")
        if label_horizon <= 0:
            raise ValueError("label_horizon must be > 0")

        self.train_size = int(train_size)
        self.valid_size = int(valid_size)
        self.test_size = int(test_size)
        self.embargo_size = int(embargo_size)
        self.label_horizon = int(label_horizon)

    @property
    def window(self) -> int:
        """Total bars consumed by one fold (train + valid + test)."""
        return self.train_size + self.valid_size + self.test_size

    def split(self, timestamps: int | Sized) -> list[Fold]:
        """Generate ``(train_idx, valid_idx, test_idx)`` folds.

        Parameters
        ----------
        timestamps:
            Either the number of observations, or any sized, ordered sequence
            (e.g. a ``pd.DatetimeIndex`` or array). Only its length / positional
            order is used.

        Returns
        -------
        list[Fold]
            One ``(train_idx, valid_idx, test_idx)`` tuple per fold; each entry
            is a ``numpy`` array of positional ``int`` indices. May be empty if
            every fold is fully purged. Raises ``ValueError`` if the timeline is
            shorter than a single fold (fail-closed).
        """
        n = timestamps if isinstance(timestamps, int) else len(timestamps)
        if n < self.window:
            raise ValueError(
                f"Not enough observations ({n}) for one fold (need at least {self.window})."
            )

        folds: list[Fold] = []
        start = 0
        while start + self.window <= n:
            train_end = start + self.train_size  # exclusive
            valid_end = train_end + self.valid_size  # exclusive
            test_end = valid_end + self.test_size  # exclusive

            raw_train: IntArray = np.arange(start, train_end)
            valid_idx: IntArray = np.arange(train_end, valid_end)
            test_idx: IntArray = np.arange(valid_end, test_end)

            eval_start_pos = int(train_end)  # first evaluation (valid) position
            purged = self._embargo(self._purge(raw_train, eval_start_pos), train_end)

            if purged.size == 0:
                logger.warning(
                    "Fold starting at index %d has zero training observations "
                    "after purging - skipped.",
                    start,
                )
                start += self.test_size
                continue

            folds.append((purged, valid_idx, test_idx))
            start += self.test_size

        return folds

    def _purge(self, train_idx: IntArray, eval_start_pos: int) -> IntArray:
        """Keep training obs whose label window ends strictly before eval start.

        An observation at position ``i`` carries a forward label spanning bars
        ``[i, i + label_horizon]``; it is dropped unless
        ``i + label_horizon < eval_start_pos`` (positional, NOT calendar-day).
        """
        idx = np.asarray(train_idx, dtype=np.int_)
        return idx[idx + self.label_horizon < int(eval_start_pos)]

    def _embargo(self, train_idx: IntArray, train_end: int) -> IntArray:
        """Drop the last ``embargo_size`` training bars before the eval window."""
        if self.embargo_size == 0 or train_idx.size == 0:
            return train_idx
        cutoff = int(train_end) - self.embargo_size
        return train_idx[train_idx < cutoff]


def default_walk_forward_splitter(n_samples: int) -> PurgedWalkForwardSplitter:
    """A sensible auto-sized walk-forward splitter for ``n_samples`` bars.

    Allocates roughly 60% train / 20% valid / 20% test with a one-bar embargo
    and one-bar label horizon -- adequate defaults for a single per-period signal
    series. These sizes are heuristics, not published constants; supply your own
    :class:`PurgedWalkForwardSplitter` for production policy.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    test = max(2, n_samples // 5)
    train = max(4, n_samples - 3 * test)
    valid = test
    return PurgedWalkForwardSplitter(
        train_size=train,
        valid_size=valid,
        test_size=test,
        embargo_size=1,
        label_horizon=1,
    )
