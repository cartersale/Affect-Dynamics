from __future__ import annotations

import numpy as np


def debounce_short_runs(x: np.ndarray, *, k: int) -> np.ndarray:
    """
    Merge runs of identical values with length <= k into neighboring runs.

    Deterministic rule:
    - If a short run is between two equal values, merge into that value.
    - Otherwise merge into the longer neighbor.
    - If only one neighbor exists (edge), merge into that neighbor.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if len(x) == 0 or k <= 0:
        return x.copy()

    out = x.copy()

    # Run-length encode
    vals = []
    starts = []
    lens = []

    i = 0
    n = len(out)
    while i < n:
        v = out[i]
        j = i + 1
        while j < n and out[j] == v:
            j += 1
        vals.append(v)
        starts.append(i)
        lens.append(j - i)
        i = j

    vals = np.array(vals)
    starts = np.array(starts)
    lens = np.array(lens)

    # Process short runs
    for r in range(len(vals)):
        if lens[r] > k:
            continue

        left = r - 1
        right = r + 1

        left_exists = left >= 0
        right_exists = right < len(vals)

        if not left_exists and not right_exists:
            continue  # single-run sequence

        if left_exists and right_exists and vals[left] == vals[right]:
            fill = vals[left]
        elif left_exists and right_exists:
            # choose longer neighbor; tie -> left
            fill = vals[left] if lens[left] >= lens[right] else vals[right]
        elif left_exists:
            fill = vals[left]
        else:
            fill = vals[right]

        s = starts[r]
        e = s + lens[r]
        out[s:e] = fill

    return out
