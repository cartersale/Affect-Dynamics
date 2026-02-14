from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray


def groupkfold_splits(
    groups: list[str], *, n_splits: int = 5, seed: int | None = None
) -> list[Split]:
    # GroupKFold is deterministic given ordering; seed unused (kept for future).
    gkf = GroupKFold(n_splits=n_splits)
    X = np.zeros((len(groups), 1))
    splits: list[Split] = []
    for tr, te in gkf.split(X, y=None, groups=groups):
        splits.append(Split(train_idx=tr, test_idx=te))
    return splits
