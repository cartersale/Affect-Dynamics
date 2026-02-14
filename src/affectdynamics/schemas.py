from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Session:
    session_id: str
    dyad_id: str
    therapist_codes: np.ndarray
    client_codes: np.ndarray
    t: np.ndarray | None
    meta: dict

    def to_parquet(self, path: Path):
        df = pd.DataFrame(
            {
                "t": self.t if self.t is not None else np.arange(len(self.therapist_codes)),
                "T": self.therapist_codes,
                "C": self.client_codes,
            }
        )
        df.attrs["session_id"] = self.session_id
        df.attrs["dyad_id"] = self.dyad_id
        df.attrs["meta"] = self.meta
        df.to_parquet(path, index=False)

    @staticmethod
    def from_parquet(path: Path) -> Session:
        df = pd.read_parquet(path)
        return Session(
            session_id=df.attrs["session_id"],
            dyad_id=df.attrs["dyad_id"],
            therapist_codes=df["T"].to_numpy(),
            client_codes=df["C"].to_numpy(),
            t=df["t"].to_numpy(),
            meta=df.attrs.get("meta", {}),
        )
