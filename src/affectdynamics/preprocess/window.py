from __future__ import annotations


def sliding_windows(n: int, *, win: int, step: int) -> list[tuple[int, int]]:
    """
    Generate sliding window indices for a sequence of length n.

    Args:
        n: The total length of the sequence.
        win: The size (length) of the sliding window.
        step: The step size (stride) to move the window forward.

    Returns:
        A list of (start, end) tuples representing the slice indices for each window.
        If the window size is larger than n, returns a single window covering the whole sequence.
    """
    if n <= 0:
        return []
    if win <= 0 or step <= 0:
        raise ValueError("win and step must be > 0")
    if win > n:
        return [(0, n)]

    out: list[tuple[int, int]] = []
    start = 0
    while start + win <= n:
        out.append((start, start + win))
        start += step
    return out
