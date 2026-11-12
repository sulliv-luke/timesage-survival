"""Vendored from sage-importance 0.0.6 (https://github.com/iancovert/sage).

MIT License, Copyright (c) 2020 Ian Covert. The full licence text is in
``LICENSE_sage.txt`` next to this file.

Sources: ``sage/imputers.py`` (sha256 17d2f4f9...ed424) and ``sage/utils.py``
(sha256 88e46f7d...fba4c) of the 0.0.6 wheel. The original TimeSAGE code used
these classes unchanged. They are copied here so the package does not depend
on sage-importance, which also pulls in matplotlib, tqdm and joblib.

The arithmetic is copied verbatim, so results stay bit-identical to the
original research implementation.
Changes:

- type annotations and docstrings added;
- ``Imputer.__init__`` no longer calls ``sage.utils.model_conversion`` and
  instead requires a callable. For a callable, ``model_conversion`` returned it
  unchanged, so behaviour is identical. TimeSAGE always passes a callable;
- only ``Imputer``, ``MarginalImputer`` and ``ImportanceTracker`` are kept.

This file is excluded from ruff so it can be diffed against the upstream
source line by line.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

ModelFn = Callable[[NDArray[Any]], NDArray[Any]]


class Imputer:
    '''Imputer base class.'''
    def __init__(self, model: ModelFn):
        if not callable(model):
            raise TypeError('model must be callable')
        self.model = model

    def __call__(self, x: NDArray[Any], S: NDArray[np.bool_]) -> NDArray[Any]:
        raise NotImplementedError


class MarginalImputer(Imputer):
    '''Marginalizing out removed features with their marginal distribution.'''
    def __init__(self, model: ModelFn, data: NDArray[Any]):
        super().__init__(model)
        self.data = data
        self.data_repeat = data
        self.samples = len(data)
        self.num_groups = data.shape[1]

        if len(data) > 1024:
            warnings.warn('using {} background samples may lead to slow '
                          'runtime, consider using <= 1024'.format(
                            len(data)), RuntimeWarning)

    def __call__(self, x: NDArray[Any], S: NDArray[np.bool_]) -> NDArray[Any]:
        # Prepare x and S.
        n = len(x)
        x = x.repeat(self.samples, 0)
        S = S.repeat(self.samples, 0)

        # Prepare samples.
        if len(self.data_repeat) != self.samples * n:
            self.data_repeat = np.tile(self.data, (n, 1))

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = self.data_repeat[~S]

        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        mean: NDArray[Any] = np.mean(pred, axis=1)
        return mean


class ImportanceTracker:
    '''For tracking feature importance using a dynamic average.'''
    def __init__(self) -> None:
        self.mean: Any = 0
        self.sum_squares: Any = 0
        self.N: Any = 0

    def update(self, scores: NDArray[Any], num_samples: NDArray[Any] | None = None) -> None:
        '''
        Update mean and sum of squares using Welford's algorithm.

        Args:
          scores: array of consisting of n samples with shape (n, dim).
          num_samples: array of size (dim,) representing the number of samples
            for each dimension. For sparse updates, with void samples
            represented by zeros.
        '''
        if num_samples is None:
            # Welford's algorithm.
            self.N += len(scores)
            diff = scores - self.mean
            self.mean += np.sum(diff, axis=0) / self.N
            diff2 = scores - self.mean
            self.sum_squares += np.sum(diff * diff2, axis=0)
        else:
            # Welford's algorithm with correction for void samples.
            assert num_samples.shape == scores.shape[1:]
            self.N = self.N + num_samples
            num_void = len(scores) - num_samples
            orig_mean = np.copy(self.mean)
            diff = scores - self.mean
            self.mean += (
                np.sum(diff, axis=0) +
                self.mean * num_void) / np.maximum(self.N, 1)
            diff2 = scores - self.mean
            self.sum_squares += (
                np.sum(diff * diff2, axis=0) -
                orig_mean * self.mean * num_void)

    @property
    def values(self) -> Any:
        return self.mean

    @property
    def var(self) -> Any:
        # print('sum_squares', self.sum_squares)
        return self.sum_squares / (np.maximum(self.N, 1) ** 2)

    @property
    def std(self) -> Any:
        return self.var ** 0.5
