from __future__ import annotations


def sliding_windows(n: int, *, win: int, step: int) -> list[tuple[int, int]]:
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
